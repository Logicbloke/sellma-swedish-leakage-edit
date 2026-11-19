"""Step 3: choose layer and alpha for the edit on the DEV split only.

Selection rule (fixed before running): minimise the no/da sentence share on dev SV prompts, subject to
  (i) dev EN prompts still answered in English at >= guardrail_en_rate, and
  (ii) dev SV prompts still answered in Swedish at >= baseline rate - 0.05.
"""
import json, time, torch
import pandas as pd
from common import CFG, RESULTS, load_model, load_jsonl, chat, generate, write_jsonl
from edit import apply_edit, revert_edit
from metrics import score_rows, summarize

tok, model = load_model()
D = torch.load(RESULTS / "directions.pt")["d_nordic"]
sets = {l: load_jsonl(f"dev_{l}.jsonl") for l in ["sv", "en"]}


def run(tag):
    res = {"config": tag}
    for l, P in sets.items():
        outs = generate(tok, model, [chat(tok, p["prompt"], CFG["system_prompt"]) for p in P])
        rows = score_rows([{**p, "output": o} for p, o in zip(P, outs)])
        write_jsonl(RESULTS / "dev_sweep" / f"{tag}_{l}.jsonl", rows)
        res.update({f"{l}:{k}": v for k, v in summarize(rows, l).items()})
    print(json.dumps(res), flush=True)
    return res


t = time.time()
table = [dict(run("baseline"), layer=None, alpha=0.0)]
for L in CFG["layers"]:
    for a in CFG["alphas"]:
        old = apply_edit(model, L, a * D[L])
        table.append(dict(run(f"L{L}_a{a}"), layer=L, alpha=a))
        revert_edit(model, L, old)
df = pd.DataFrame(table)
df.to_csv(RESULTS / "dev_sweep.csv", index=False)
base_sv = df.loc[0, "sv:sv_answer_rate"]
ok = df[(df.layer.notna()) & (df["en:en_answer_rate"] >= CFG["guardrail_en_rate"])
        & (df["sv:sv_answer_rate"] >= base_sv - 0.05)]
if ok.empty:
    raise SystemExit("No edit config passed the guardrails; see results/dev_sweep.csv")
best = ok.sort_values(["sv:nordic_sent_share", "alpha"]).iloc[0]
sel = {"layer": int(best.layer), "alpha": float(best.alpha), "dev_nordic_sent_share": float(best["sv:nordic_sent_share"]),
       "baseline_dev_nordic_sent_share": float(df.loc[0, "sv:nordic_sent_share"]), "n_eligible": len(ok)}
json.dump(sel, open(RESULTS / "selected_edit.json", "w"), indent=2)
print("SELECTED", sel, f"{time.time() - t:.0f}s")
