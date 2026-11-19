# Manual audit of the primary metric (dev split, sentences flagged no/da by lingua)

Done by hand (Claude Code, reviewed by the author) on the dev outputs, before looking at any test results.
Labels: G = genuine Norwegian/Danish leakage, B = borderline, F = false positive.

## Baseline (17 flagged sentences, from 7 of 40 answers)

| id | lingua | sentence | label | evidence |
|---|---|---|---|---|
| dev-45-2 | DA | Verificering: Verifikationer | B | "verificering" is Danish-like; "verifikationer" is valid Swedish |
| dev-45-3 | DA | Om vi inte har integritet, kommer ni att ha problemer med att ta | G | problemer (sv: problem) |
| dev-3-4 | DA | Kvalitets- og kostsforhold: Många forskellige kvalitetssystemer … | G | og, forskellige, -systemer |
| dev-3-4 | DA | Kostsforhold: Inre inflationen kan være en del av kostsforholdet | G | være |
| dev-9-1 | DA | Medlemmer kan skriva in meddelanden (×3) | G | medlemmer (sv: medlemmar) |
| dev-9-1 | DA | Medlemmer kan skriv in meddelanden | G | medlemmer |
| dev-9-1 | DA | Fokus på en specifik tid eller sted | G | sted (sv: ställe/plats) |
| dev-29-4 | NY | Schachrollen spelar varierende rolle i vardagen, baseret på de mest populære … | G | varierende, rolle, baseret, populære |
| dev-17-2 | NY | Kortstyrande av produkt | F | malformed Swedish, no no/da forms |
| dev-17-2 | BO | Kvalitetskontroll och kontekstforskning | G (weak) | kontekst (sv: kontext) |
| dev-17-4 | DA | Fokus på investeringer og finansiering | G | investeringer, og |
| dev-17-4 | DA | Viden om ekonomiske situationer | G | viden, ekonomiske |
| dev-17-4 | DA | Kvalitetsanalyser av flerere handlere | G | flerere, handlere |
| dev-17-4 | BO | Kontrollering av risker | F | "risker" is Swedish; "kontrollering" is non-standard but not no/da |
| dev-17-4 | DA | Finansiering av projekter | G | projekter (sv: projekt) |

Precision: 14 G / 17 (82%), or 15/17 (88%) if the borderline case counts.

## Selected edit L8, α=8 (1 flagged sentence, 1 of 40 answers)

| id | lingua | sentence | label |
|---|---|---|---|
| dev-45-1 | NY | Kontrollera krypteringsuttrykk | G (uttrykk = no "expression") |

## Caveats found
- **Clustering:** baseline flags come from 7 answers, and one answer (dev-9-1) contributes 5 sentences. Per-answer
  bootstrapping (used for the CIs) handles this better than the pooled sentence share.
- **Recall gap:** the metric misses word-level leakage inside sentences that lingua still labels Swedish. Edited
  outputs still contain forms like "systemer", "Verifisera" and "behovar". The metric measures *sentence-level*
  switches, not overall Swedish correctness.
- **Degenerate outputs:** at L16, α=16 the model emits garbage ("qttos de qttos …"). Lingua labels some of it
  no/da, so that config's "50% no/da" is an artefact, not leakage.
