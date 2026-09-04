# Adaptive model-poisoning attacks: ALIE (Baruch et al. 2019), IPM (Fang et al. 2020)
import torch
from torch.distributions import Normal

_STD = Normal(0.0, 1.0)

def _flatten(updates):
    keys = list(updates[0].keys())
    flat = torch.stack([torch.cat([u[k].float().view(-1) for k in keys]) for u in updates])
    return flat, keys

def _unflatten(vec, template):
    out, idx = {}, 0
    for k in template.keys():
        ne = template[k].numel()
        out[k] = vec[idx:idx + ne].view(template[k].shape).to(template[k].dtype)
        idx += ne
    return out

def alie_attack(benign_updates, num_malicious, num_sampled_total):
    """A Little Is Enough: mal = mu - z*sigma, z = Phi^{-1}((n/2 - f)/(n - f)).
    Population std keeps it well-defined with a single benign reference."""
    flat, _ = _flatten(benign_updates)
    mu, sigma = flat.mean(0), flat.std(0, correction=0)
    n, f = num_sampled_total, num_malicious
    s = min(max((n / 2 - f) / max(n - f, 1), 1e-3), 1 - 1e-3)
    z = _STD.icdf(torch.tensor(s)).item()
    return [_unflatten(mu - z * sigma, benign_updates[0]) for _ in range(num_malicious)]

def ipm_attack(benign_updates, num_malicious):
    """Inner Product Manipulation: negated, norm-matched benign consensus."""
    flat, _ = _flatten(benign_updates)
    mu = flat.mean(0)
    eps = flat.norm(dim=1).mean() / (mu.norm() + 1e-8)
    return [_unflatten(-eps * mu, benign_updates[0]) for _ in range(num_malicious)]

def generate_adaptive_attacks(benign_updates, attack, num_malicious, num_sampled_total):
    if attack == "alie": return alie_attack(benign_updates, num_malicious, num_sampled_total)
    if attack == "ipm":  return ipm_attack(benign_updates, num_malicious)
    raise ValueError(f"Unknown adaptive attack: {attack}")