import torch
import numpy as np
import copy

from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import coordinate_wise_median, krum_aggregation
from attacks import generate_adaptive_attacks

NUM_CLIENTS = 20
NUM_ROUNDS = 20
CORRUPTION = 0.3
PARTICIPATION = 0.1
ATTACK = "ipm"
SEED = 42

METHODS = ["Median", "Krum"]


def assumed_f(n, corr):
    return max(0, int(n * corr))


def run_trace(method, client_loaders, test_loader):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))
    num_sampled = max(1, int(NUM_CLIENTS * PARTICIPATION))
    f = assumed_f(num_sampled, CORRUPTION)

    model = SimpleCNN().to(device)
    worker = SimpleCNN().to(device)

    print(f"\n=== {method} | {ATTACK} | beta={CORRUPTION} | p={PARTICIPATION} "
          f"(n={num_sampled}) | seed={SEED} | {NUM_ROUNDS} rounds ===")
    if method == "Krum":
        precondition_ok = num_sampled > 2 * f + 2
        print(f"    Krum precondition n>2f+2: n={num_sampled}, f={f}, "
              f"required n>{2*f+2} -> {'SATISFIED' if precondition_ok else 'VIOLATED'}")

    trajectory = []
    for r in range(1, NUM_ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, num_sampled, replace=False)
        current = model.state_dict()
        mal_ids = [c for c in sampled if c in byz]
        ben_ids = [c for c in sampled if c not in byz]

        ben_states = []
        for cid in ben_ids:
            worker.load_state_dict(current)
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01)
            ben_states.append({k: v.detach().clone() for k, v in st.items()})

        if mal_ids and ben_states:
            bu = [{k: s[k].float() - current[k].float() for k in current} for s in ben_states]
            mu = generate_adaptive_attacks(bu, ATTACK, len(mal_ids), num_sampled_total=len(sampled))
            mal_states = [{k: current[k].float() + u[k] for k in current} for u in mu]
        elif mal_ids:
            mal_states = []
            for cid in mal_ids:
                worker.load_state_dict(current)
                st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type="sign_flip")
                mal_states.append({k: v.detach().clone() for k, v in st.items()})
        else:
            mal_states = []

        states = ben_states + mal_states

        if method == "Median":
            agg = coordinate_wise_median(states)
        elif method == "Krum":
            agg = krum_aggregation(states, num_byzantine=f)
        model.load_state_dict(agg)

        w = torch.cat([v.float().view(-1) for v in model.state_dict().values()])
        acc = evaluate(model, test_loader)
        trajectory.append(acc)
        print(f"R{r:2d} mal={len(mal_ids)}/{len(sampled)} acc={acc:5.2f}% "
              f"NaN={torch.isnan(w).sum().item()} Inf={torch.isinf(w).sum().item()} |w|={w.norm():.2f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
    preds = torch.cat(preds)
    u, c = torch.unique(preds, return_counts=True)
    hist = dict(zip(u.tolist(), c.tolist()))
    print("predicted-class histogram:", hist)
    print(f"final |w|={w.norm():.2f}, final acc={trajectory[-1]:.2f}%, "
          f"classes predicted={len(hist)}")

    return trajectory, hist


if __name__ == "__main__":
    np.random.seed(123)
    client_loaders, test_loader, _ = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    results = {}
    for method in METHODS:
        traj, hist = run_trace(method, client_loaders, test_loader)
        results[method] = dict(trajectory=traj, final_hist=hist)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, r in results.items():
        collapsed = len(r["final_hist"]) <= 2
        print(f"{method:<8} final_acc={r['trajectory'][-1]:6.2f}%  "
              f"classes_predicted={len(r['final_hist'])}  "
              f"collapsed={'YES' if collapsed else 'no'}")
