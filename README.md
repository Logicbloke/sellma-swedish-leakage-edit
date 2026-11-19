# Can a one-vector weight edit stop a small LLM from slipping into Norwegian/Danish when answering in Swedish?

A small, honest test of one claim from *Sparks of Weight-Space Programming* (Takehi et al., 2026): that tiny targeted
weight edits can remove a failure mode, without adding capability. Candidate assignment for SELLMA.

**Report:** [`report/note.md`](report/note.md) (and `report/note.pdf`). **AI usage:** [`AI_USAGE.md`](AI_USAGE.md).

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and about 2 h on 8 CPU cores (no GPU needed).

```bash
uv sync          # Python 3.12, torch 2.5.1 (CPU), transformers 4.46.3, lingua
make all         # data -> pilot -> direction -> dev sweep -> test evaluation -> figures
```

Individual steps: `make data | pilot | direction | sweep | evaluate | plots`.

## Configuration

`config.yaml` records everything that matters: the model (`Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`), greedy decoding (so no sampling seeds are involved), seed 0,
`max_new_tokens=96`, the system prompt, the layer/α search grid, the guardrail, and the random-direction seeds.

## Layout

| Path | Contents |
|---|---|
| `src/make_data.py` | Builds all prompt sets (48 topics × 5 templates in sv/en/no/da), en/ett pairs, and the Wikipedia perplexity text |
| `src/pilot.py` | Go/no-go: how often does the unedited model drift? (dev split) |
| `src/direction.py` | Difference-of-means directions SV−(NO∪DA) and SV−EN (direction split only) |
| `src/edit.py` | The edit: a bias vector on one layer's `mlp.down_proj` |
| `src/sweep.py` | Layer/α selection on the dev split, with a fixed selection rule |
| `src/evaluate.py` | Final held-out test evaluation of all conditions and metrics, plus sanity checks |
| `src/metrics.py`, `src/common.py` | Language ID, leakage metrics, perplexity, agreement accuracy |
| `data/` | All evaluation data (committed). Splits are by topic: direction 20 topics, dev 8, test 20, no overlap |
| `results/` | Raw generations (`pilot/`, `dev_sweep/`, `test/`), summaries (`*.csv`, `*.json`), the edit itself (`edit_nordic.pt`), figures |

## Data provenance

The prompts are synthetic (templates × topic names written for this project). The perplexity text is the lead
sections of 10 Swedish and 10 English Wikipedia articles (CC BY-SA 4.0; source URLs are in `data/ppl_*.jsonl`).
