"""Score a list of generations for a given prompt language."""
import numpy as np
from common import detect_lang, leak_stats, sentence_nordic_share, bootstrap_ci

REFUSAL = ("I'm sorry", "I am sorry", "I can't assist", "Jag är ledsen", "Tyvärr kan jag inte")


def score_rows(rows):
    """Add per-row metrics in place; rows need 'output'."""
    for r in rows:
        o = r["output"]
        r["detected"] = detect_lang(o)
        r["leak_hits"], r["n_words"] = leak_stats(o)
        r["nordic_sents"], r["n_sents"] = sentence_nordic_share(o)
        r["refusal"] = any(s in o for s in REFUSAL)
    return rows


def summarize(rows, lang):
    """Aggregate metrics. For per-prompt CIs, the sentence share is computed per answer (answers without
    any scorable sentence count as 0) and bootstrapped over prompts."""
    n = len(rows)
    out = {"n": n,
           f"{lang}_answer_rate": float(np.mean([r["detected"] == lang for r in rows])),
           "refusal_rate": float(np.mean([r["refusal"] for r in rows])),
           # degenerate (empty/short) answers score 0 leakage, so report length alongside
           "mean_words": float(np.mean([r["n_words"] for r in rows])),
           "mean_sents": float(np.mean([r["n_sents"] for r in rows]))}
    if lang == "sv":
        per = [r["nordic_sents"] / r["n_sents"] if r["n_sents"] else 0.0 for r in rows]
        m, lo, hi = bootstrap_ci(per)
        out.update(nordic_sent_share=m, nordic_sent_ci=[lo, hi],
                   nordic_sent_pooled=sum(r["nordic_sents"] for r in rows) / max(1, sum(r["n_sents"] for r in rows)),
                   leak_per_100w=100 * sum(r["leak_hits"] for r in rows) / max(1, sum(r["n_words"] for r in rows)),
                   any_nordic_rate=float(np.mean([r["nordic_sents"] > 0 or r["detected"] in ("no", "da") for r in rows])))
        m, lo, hi = bootstrap_ci([r["detected"] == "sv" for r in rows])
        out["sv_answer_ci"] = [lo, hi]
    return out
