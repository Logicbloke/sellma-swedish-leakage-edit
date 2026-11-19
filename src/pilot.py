"""Step 0 (go/no-go): how often does the unedited model drift out of Swedish on Swedish prompts?

Uses the DEV split only, under three system-prompt settings.
"""
import collections, json, time
from common import CFG, RESULTS, load_model, load_jsonl, chat, generate, detect_lang, write_jsonl

tok, model = load_model()
prompts = load_jsonl("dev_sv.jsonl")
summary = {}
for setting in ["default", "en_assistant", "empty"]:
    t = time.time()
    outs = generate(tok, model, [chat(tok, p["prompt"], setting) for p in prompts])
    langs = [detect_lang(o) for o in outs]
    rows = [{**p, "setting": setting, "output": o, "detected": l} for p, o, l in zip(prompts, outs, langs)]
    write_jsonl(RESULTS / "pilot" / f"{setting}.jsonl", rows)
    c = collections.Counter(langs)
    summary[setting] = {"n": len(langs), "sv_rate": c["sv"] / len(langs), "counts": dict(c.most_common())}
    print(setting, summary[setting], f"{time.time() - t:.0f}s", flush=True)
json.dump(summary, open(RESULTS / "pilot_summary.json", "w"), indent=2)
