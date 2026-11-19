"""Shared helpers: config, model loading, batched greedy generation, language ID, scoring."""
import json, pathlib, random, re
import numpy as np, torch, yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))

SYSTEM_PROMPTS = {
    "default": None,  # Qwen2.5 chat template inserts its own default system prompt
    "en_assistant": "You are a helpful assistant. Answer concisely.",
    "empty": "",
}


def seed_all(seed=CFG["seed"]):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_jsonl(name):
    return [json.loads(l) for l in open(DATA / name, encoding="utf-8")]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_model():
    seed_all()
    torch.set_num_threads(CFG["threads"])
    kw = dict(revision=CFG["model_revision"])
    tok = AutoTokenizer.from_pretrained(CFG["model_id"], **kw)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], torch_dtype=getattr(torch, CFG["dtype"]), **kw)
    model.eval()
    # silence sampling-param warnings: we decode greedily
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return tok, model


def chat(tok, user, system="default", prefix=None):
    """Render a chat prompt. `prefix` is prepended to the user turn (used for the prompting baseline)."""
    sp = SYSTEM_PROMPTS[system]
    msgs = [] if sp is None else [{"role": "system", "content": sp}]
    msgs.append({"role": "user", "content": f"{prefix} {user}" if prefix else user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, texts, max_new_tokens=None, batch_size=None):
    max_new_tokens = max_new_tokens or CFG["gen"]["max_new_tokens"]
    batch_size = batch_size or CFG["gen"]["batch_size"]
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


# ---------------- language ID ----------------
_DET = None
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def detect_lang(text):
    """ISO-639-1 code of the dominant language; 'und' for too-short text, 'zh' if mostly CJK."""
    global _DET
    from lingua import LanguageDetectorBuilder
    if _DET is None:
        _DET = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    t = text.strip()
    if len(t) < CFG["min_chars_for_langid"]:
        return "und"
    if len(CJK.findall(t)) > 0.3 * len(t):
        return "zh"
    lang = _DET.detect_language_of(t)
    if lang is None:
        return "und"
    code = lang.iso_code_639_1.name.lower()
    return "no" if code in ("nb", "nn") else code


def bootstrap_ci(xs, n=2000, seed=0):
    xs = np.asarray(xs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(xs, size=(n, len(xs)), replace=True).mean(1)
    return float(xs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------- likelihood metrics ----------------
@torch.no_grad()
def seq_logprob(tok, model, texts):
    """Sum of token log-probs of each text (no chat template), batched."""
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True)
    tok.padding_side = "left"
    logits = model(**enc).logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    lp = torch.log_softmax(logits, -1).gather(-1, tgt[..., None])[..., 0]
    mask = enc.attention_mask[:, 1:].float()
    return (lp * mask).sum(1).tolist(), mask.sum(1).tolist()


def perplexity(tok, model, texts):
    tot_lp, tot_n = 0.0, 0.0
    for t in texts:
        lp, n = seq_logprob(tok, model, [t])
        tot_lp += lp[0]; tot_n += n[0]
    return float(np.exp(-tot_lp / tot_n))


def agreement_acc(tok, model, pairs):
    """Fraction of en/ett minimal pairs where the grammatical sentence gets higher log-prob."""
    good, _ = seq_logprob(tok, model, [p["good"] for p in pairs])
    bad, _ = seq_logprob(tok, model, [p["bad"] for p in pairs])
    return [int(g > b) for g, b in zip(good, bad)]


# ---------------- Norwegian/Danish leakage in Swedish text ----------------
# Pre-registered lexicon (fixed after the pilot, before any dev/test evaluation): common Norwegian-Bokmål /
# Danish word forms that are NOT standard Swedish words. Swedish/no/da homographs (e.g. "kan", "eller",
# "mig", "efter", "blev", "problemet",
# "er" = Swedish 'your' (pl.), "andre" = Swedish 'second/other' (removed after code review, before any test eval)) are deliberately excluded.
LEAK_WORDS = set("""
ikke ikkje også og være vært været hvordan hvad hva hvis hvor hvilke hvilken hvem hvorfor noen nogen nogle
mange meget mye bare til fra jeg meg dere deres hun kunne gjøre gjør gøre gør mellom mellem måte måde samfunn
samfund sosial sosiale sociale problemer betydning betydningen viktige eksempel eksempler kjent kendt hjelpe
hjælpe hjelp hjælp ble bliver blitt etter skal vil nok litt lidt enn dette disse uten uden mennesker nye alle
ofte mere hverdag hverdagen rolle følge bruke bruge brukes bruges lære tilbake tilbage derfor sammen
""".split())
WORD = re.compile(r"[a-zåäöæøéü]+")


def leak_stats(text):
    """(leak word hits, total words) for one text."""
    words = WORD.findall(text.lower())
    return sum(w in LEAK_WORDS for w in words), len(words)


_SDET = None


def sentence_nordic_share(text):
    """Share of sentences (>= min_chars) that lingua, restricted to {sv, nb, nn, da, en}, labels nb/nn/da.
    Returns (n_nordic, n_sentences)."""
    global _SDET
    from lingua import Language, LanguageDetectorBuilder
    if _SDET is None:
        _SDET = LanguageDetectorBuilder.from_languages(
            Language.SWEDISH, Language.BOKMAL, Language.NYNORSK, Language.DANISH, Language.ENGLISH
        ).with_preloaded_language_models().build()
    sents = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) >= CFG["min_chars_for_langid"]]
    labs = [_SDET.detect_language_of(s) for s in sents]
    nordic = sum(l is not None and l.name in ("BOKMAL", "NYNORSK", "DANISH") for l in labs)
    return nordic, len(sents)
