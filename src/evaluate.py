"""Step 4: final evaluation on the held-out TEST split (touched once, after the edit was selected on dev).

Conditions
  baseline     unchanged model
  prompt       instruction prefix in the user turn (cfg.prompt_baseline), no weight change
  edit_nordic  selected edit: alpha * d_nordic[L] as down_proj bias at layer L
  edit_en      same layer, same vector norm, but along d_en (SV - EN) -> does the specific direction matter?
  random_s     same layer, same norm, random direction (3 seeds)
Metrics
  target       no/da sentence share in answers to SV prompts (+ lexicon leak rate, Swedish answer rate)
  capability   en/ett agreement minimal pairs (log-prob), Swedish perplexity (Wikipedia leads)
  regression   EN prompts answered in English, English perplexity, NO prompts answered in Norwegian
"""
import json, time, torch
import numpy as np, pandas as pd
from common import (CFG, RESULTS, load_model, load_jsonl, chat, generate, write_jsonl, perplexity,
                    agreement_acc, bootstrap_ci)
from edit import apply_edit, revert_edit, random_like, save_edit, load_edit
from metrics import score_rows, summarize

OUT = RESULTS / "test"
tok, model = load_model()
dirs = torch.load(RESULTS / "directions.pt")
sel = json.load(open(RESULTS / "selected_edit.json"))
L, alpha = sel["layer"], sel["alpha"]
v_nordic = alpha * dirs["d_nordic"][L]
v_en = dirs["d_en"][L] / dirs["d_en"][L].norm() * v_nordic.norm()
save_edit(RESULTS / "edit_nordic.pt", L, v_nordic)

test = {l: load_jsonl(f"test_{l}.jsonl") for l in ["sv", "en", "no"]}
pairs = load_jsonl("agreement_pairs.jsonl")
ppl_txt = {l: [r["text"] for r in load_jsonl(f"ppl_{l}.jsonl")] for l in ["sv", "en"]}


def gens(langs, prefix=None):
    out = {}
    for l in langs:
        P = test[l]
        o = generate(tok, model, [chat(tok, p["prompt"], CFG["system_prompt"], prefix) for p in P])
        out[l] = [{**p, "output": x} for p, x in zip(P, o)]
    return out


# ---- sanity checks (fast, first 10 SV test prompts) ----
probe = [chat(tok, p["prompt"], CFG["system_prompt"]) for p in test["sv"][:10]]
base_probe = generate(tok, model, probe)
enc = tok(probe, return_tensors="pt", padding=True)
with torch.no_grad():
    base_logits = model(**enc).logits
    old = apply_edit(model, L, torch.zeros_like(v_nordic))
    zero_logits = model(**enc).logits
    revert_edit(model, L, old)
assert torch.allclose(base_logits, zero_logits, atol=1e-4), "alpha=0 edit must reproduce baseline logits"
old = apply_edit(model, L, v_nordic); mem = generate(tok, model, probe); revert_edit(model, L, old)
old = load_edit(model, RESULTS / "edit_nordic.pt"); disk = generate(tok, model, probe); revert_edit(model, L, old)
assert mem == disk, "saved edit must reproduce in-memory edit"
print("sanity checks passed", flush=True)

conditions = [("baseline", None, None, ["sv", "en", "no"]),
              # the instruction is Swedish-specific, so it is only added to SV prompts (on EN/NO it would
              # tell the model to answer in the wrong language and fake a regression); EN/NO = baseline there
              ("prompt", None, CFG["prompt_baseline"], ["sv"]),
              ("edit_nordic", v_nordic, None, ["sv", "en", "no"]),
              ("edit_en", v_en, None, ["sv", "en", "no"])]
conditions += [(f"random_{s}", random_like(v_nordic, s), None, ["sv", "en"]) for s in CFG["random_seeds"]]

summary = []
for name, vec, prefix, langs in conditions:
    t = time.time()
    old = apply_edit(model, L, vec) if vec is not None else None
    res = {"condition": name}
    for l, rows in gens(langs, prefix).items():
        score_rows(rows)
        write_jsonl(OUT / f"{name}_{l}.jsonl", rows)
        res.update({f"{l}:{k}": v for k, v in summarize(rows, l).items()})
    if prefix is None:  # likelihood metrics don't depend on the prompt prefix
        acc = agreement_acc(tok, model, pairs)
        m, lo, hi = bootstrap_ci(acc)
        res.update({"agreement_acc": m, "agreement_ci": [lo, hi],
                    "ppl_sv": perplexity(tok, model, ppl_txt["sv"]), "ppl_en": perplexity(tok, model, ppl_txt["en"])})
    if old is not None:
        revert_edit(model, L, old)
    summary.append(res)
    print(json.dumps(res), f"{time.time() - t:.0f}s", flush=True)

json.dump({"selected_edit": sel, "conditions": summary}, open(RESULTS / "test_summary.json", "w"), indent=2)


# ---- paired comparisons vs baseline on the primary metric (bootstrap over prompts) ----
def per_prompt(name):
    rows = [json.loads(x) for x in open(OUT / f"{name}_sv.jsonl")]
    return np.array([r["nordic_sents"] / r["n_sents"] if r["n_sents"] else 0.0 for r in rows])


base = per_prompt("baseline")
rng = np.random.default_rng(0)
paired = {}  # boot_tail_frac_ge_0 is descriptive, not a calibrated p-value
for name, *_ in conditions[1:]:
    diff = per_prompt(name) - base
    boots = rng.choice(diff, size=(5000, len(diff))).mean(1)
    paired[name] = {"mean_diff": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                    "boot_tail_frac_ge_0": float((boots >= 0).mean())}
json.dump(paired, open(RESULTS / "test_paired.json", "w"), indent=2)
print(json.dumps(paired, indent=1))
pd.json_normalize(summary).to_csv(RESULTS / "test_summary.csv", index=False)
