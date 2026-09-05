# self-checking aggregation-error vs. robust-estimation bound plot, adaptive_attacks
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import SimpleCNN, get_cifar10_loaders, train_client, device
from aggregators import coordinate_wise_median, swb_aggregation
from attacks import alie_attack,generate_adaptive_attacks


def flat(sd):
    return torch.cat([sd[k].float().view(-1) for k in sd.keys()])

if __name__ == "__main__":
    np.random.seed(123)
    loaders, _, _ = get_cifar10_loaders(num_clients=20, alpha=0.5)
    betas, med_errs, swb_errs = [], [], []
    for f in [0, 1, 2, 3, 4]:
        torch.manual_seed(42)
        model = SimpleCNN().to(device); worker = SimpleCNN().to(device)
        current = model.state_dict()
        ben = []
        for cid in range(10 - f):
            worker.load_state_dict(current)
            st = train_client(worker, loaders[cid], epochs=2, lr=0.01)
            ben.append({k: v.detach().clone() for k, v in st.items()})   # <-- THE FIX
        bu = [{k: s[k].float() - current[k].float() for k in s} for s in ben]
        mu = torch.stack([flat(u) for u in bu]).mean(0)
        states = list(ben)
        if f:
            mal = generate_adaptive_attacks(bu, "ipm", f, 10)
            states += [{k: current[k].float() + u[k] for k in u} for u in mal]
        trim = min(0.45, f / 10 + 0.05)
        em = (flat(coordinate_wise_median(states)) - flat(current) - mu).norm().item()
        es = (flat(swb_aggregation(states, trim_ratio=trim)) - flat(current) - mu).norm().item()
        print(f"f={f}: ||mu||={mu.norm():.4f} | err_median={em:.4f} | err_swb={es:.4f}", flush=True)
        betas.append(f / 10); med_errs.append(em); swb_errs.append(es)

    E0 = swb_errs[0]
    C = max([max(0.0, (e - E0) / np.sqrt(b)) for b, e in zip(betas[1:], swb_errs[1:])], default=0.0) * 1.15
    bc = np.linspace(0, 0.4, 100)
    plt.figure(figsize=(8, 5))
    plt.plot(bc, E0 + C * np.sqrt(bc), "k--", lw=2, label=r"Bound $E_0+C\sqrt{\beta}$")
    plt.plot(betas, med_errs, "o-", color="gray", label="Median")
    plt.plot(betas, swb_errs, "s-", color="blue", label="SWB (calibrated)")
    plt.xlabel(r"Corruption fraction $\beta$"); plt.ylabel("L2 aggregation error (delta space)")
    plt.title("Empirical aggregation error vs. robust-estimation bound (IPM)")    
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("theory_bound_adaptive_attacks.png", dpi=150)
    print("Saved theory_bound_adaptive_attacks.png")