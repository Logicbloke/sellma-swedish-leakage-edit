"""Step 1: difference-of-means language directions from the DIRECTION split only.

Residual stream after each decoder layer, at the last prompt position (where the answer starts).
  d_nordic[L] = mean(SV) - mean(NO ∪ DA)   -> main edit ("be Swedish, not Norwegian/Danish")
  d_en[L]     = mean(SV) - mean(EN)        -> ablation (the plan's original direction)
"""
import torch
from common import CFG, RESULTS, load_model, load_jsonl, chat


@torch.no_grad()
def last_token_resid(tok, model, prompts, bs=20):
    texts = [chat(tok, p["prompt"], CFG["system_prompt"]) for p in prompts]
    acts = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True)  # left padding: last position is real
        hs = model(**enc, output_hidden_states=True).hidden_states   # (n_layers+1) x [B, T, D]
        acts.append(torch.stack([h[:, -1] for h in hs[1:]], 1))      # [B, n_layers, D]; index L = after layer L
    return torch.cat(acts).float()


if __name__ == "__main__":
    tok, model = load_model()
    a = {l: last_token_resid(tok, model, load_jsonl(f"dir_{l}.jsonl")) for l in ["sv", "no", "da", "en"]}
    mu = {l: v.mean(0) for l, v in a.items()}
    d_nordic = mu["sv"] - torch.cat([a["no"], a["da"]]).mean(0)
    d_en = mu["sv"] - mu["en"]
    # separability diagnostic: projection of each language onto d_nordic, per layer
    for L in range(d_nordic.shape[0]):
        u = d_nordic[L] / d_nordic[L].norm()
        proj = {l: round(float((v[:, L] @ u).mean()), 1) for l, v in a.items()}
        print(L, f"|d_nordic|={d_nordic[L].norm():.1f} |d_en|={d_en[L].norm():.1f} "
                 f"|resid_sv|={a['sv'][:, L].norm(dim=-1).mean():.1f}", proj)
    torch.save({"d_nordic": d_nordic, "d_en": d_en}, RESULTS / "directions.pt")
