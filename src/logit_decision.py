"""Closed-set yes/no decision using the TAD token-set definition."""

import math


def answer_token_ids(tokenizer):
    groups = []
    for word in ("yes", "no"):
        ids = set()
        for prefix in ("", " "):
            for variant in (word, word.title(), word.upper()):
                try:
                    encoded = tokenizer.encode(prefix + variant, add_special_tokens=False)
                except TypeError:
                    # Kimi's custom tokenizer uses explicit BOS/EOS flags.
                    encoded = tokenizer.encode(prefix + variant, bos=False, eos=False)
                ids.update(encoded)
        if not ids:
            raise ValueError(f"No token IDs for {word!r} in tokenizer")
        groups.append(sorted(ids))
    if set(groups[0]) & set(groups[1]):
        raise ValueError("Yes/no token ID groups overlap")
    return groups


def decide(logits, tokenizer):
    """Return a verdict and pooled raw logits for the two answer groups."""
    import torch
    yes_ids, no_ids = answer_token_ids(tokenizer)
    scores = logits.float().reshape(-1)
    yes = torch.logsumexp(scores[yes_ids], dim=0).item()
    no = torch.logsumexp(scores[no_ids], dim=0).item()
    if not math.isfinite(yes) or not math.isfinite(no):
        raise ValueError("Non-finite answer logit")
    return {"prediction": "yes" if yes > no else "no", "yes_logit": yes,
            "no_logit": no, "logit_margin": yes - no,
            "yes_token_ids": yes_ids, "no_token_ids": no_ids}
