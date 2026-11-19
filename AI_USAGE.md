# AI usage

- **Claude Code (Claude Opus 5.5)** was used as a pair-programmer: it wrote most of the code in `src/`, generated
  the synthetic prompt templates and topic lists (Swedish, English, Norwegian Bokmål, Danish) and the en/ett noun
  lists, ran the experiments, and drafted the report. Design decisions (the hypothesis, retargeting from whole-answer
  drift to Norwegian/Danish leakage after the pilot, the selection rule, the controls) were discussed and approved
  step by step by the author.
- A separate Claude subagent did an independent code review before the test run. It found that the
  prompting baseline's instruction was also being added to EN/NO prompts, which was fixed before any test generation.
- Model outputs were spot-checked by hand (see the report). All numbers in the report come from files in `results/`.
- No AI system was used as an evaluator or judge; all metrics are automatic (lingua language ID, a fixed lexicon,
  log-probabilities).
