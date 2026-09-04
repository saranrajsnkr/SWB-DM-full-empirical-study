#Naive Attacks Validation (20% corruption)
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (fed_avg, coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache

NUM_CLIENTS = 20
NUM_ROUNDS = 10
CORRUPTION = 0.2

ATTACKS = ["label_flip", "sign_flip", "gaussian"]
METHODS = ["FedAvg", "Median", "Krum", "Bulyan", "FLTrust", "SWB", "SWB-DM"]


def assumed_f(n_sample, corruption=CORRUPTION):
    """Fixed, defender-side Byzantine assumption -- computed from the
    actual sample size and corruption rate, never hardcoded. This is what
    prevents Krum/Bulyan's own preconditions (n > 2f+2, n >= 4f+3) from
    silently breaking whenever n_sample or corruption changes later
    (e.g. under a low-participation condition)."""
    return max(0, int(n_sample * corruption))


def run(method, attack, client_loaders, test_loader, root_loader, counts, seed=42, debug_trust=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))  # clients 0-3 are Byzantine
    n_sample = 11 if method == "Bulyan" else 10       # Bulyan precondition n >= 4f+3
    f = assumed_f(n_sample)                           # dynamic, not hardcoded

    global_model = SimpleCNN().to(device)
    worker = SimpleCNN().to(device)
    server_model = SimpleCNN().to(device) if method == "FLTrust" else None
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb_aggregation) if method == "SWB-DM" else None

    for r in range(1, NUM_ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
        states, s_counts, ids = [], [], []
        for cid in sampled:
            att = attack if cid in byz else None
            worker.load_state_dict(global_model.state_dict())
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
            states.append({k: v.detach().clone() for k, v in st.items()})
            s_counts.append(counts[cid])
            ids.append(cid)

        if cache is not None:
            global_model.load_state_dict(cache.step(ids, states))
        elif method == "FedAvg":
            global_model = fed_avg(global_model, states, s_counts)
        elif method == "Median":
            global_model.load_state_dict(coordinate_wise_median(states))
        elif method == "Krum":
            global_model.load_state_dict(krum_aggregation(states, num_byzantine=f))
        elif method == "Bulyan":
            global_model.load_state_dict(bulyan_aggregation(states, num_byzantine=f))
        elif method == "FLTrust":
            server_model.load_state_dict(global_model.state_dict())
            srv = train_client(server_model, root_loader, epochs=1, lr=0.01)
            current_global = global_model.state_dict()
            if debug_trust:
                _print_trust_debug(states, ids, byz, srv, current_global, r)
            global_model.load_state_dict(fltrust_aggregation(states, srv, current_global))
        elif method == "SWB":
            global_model.load_state_dict(swb_aggregation(states))

    return evaluate(global_model, test_loader)


def _print_trust_debug(states, ids, byz, server_update, global_state, round_idx):
    """Recompute FLTrust's cosine-similarity trust scores just for
    inspection, so we can directly verify Byzantine clients are actually
    being down-weighted rather than inferring it from final accuracy alone."""
    keys = states[0].keys()
    client_updates = torch.stack(
        [torch.cat([(s[k].float() - global_state[k].float()).view(-1) for k in keys]) for s in states]
    )
    server_flat = torch.cat(
        [(server_update[k].float() - global_state[k].float()).view(-1) for k in keys]
    )
    cos_sim = torch.nn.functional.cosine_similarity(client_updates, server_flat.unsqueeze(0), dim=1)
    trust = torch.relu(cos_sim)
    weights = trust / trust.sum() if trust.sum() > 1e-8 else torch.ones_like(trust) / len(trust)

    byz_w = [round(weights[i].item(), 4) for i, cid in enumerate(ids) if cid in byz]
    hon_w = [round(weights[i].item(), 4) for i, cid in enumerate(ids) if cid not in byz]
    if round_idx == 1 or round_idx == NUM_ROUNDS:
        print(f"    [trust debug] round {round_idx}: "
              f"Byzantine weights={byz_w} (n={len(byz_w)}, mean={np.mean(byz_w) if byz_w else 0:.4f}) | "
              f"Honest weights mean={np.mean(hon_w):.4f}")


def run_extended_swb_dm(attack, client_loaders, test_loader, num_rounds=30, checkpoint_every=2, seed=42):
    """Distinguish 'still warming up' from 'genuinely suppressed by attack'
    for SWB-DM by extending well past the original 10-round measurement."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))
    n_sample = 10
    global_model = SimpleCNN().to(device)
    worker = SimpleCNN().to(device)
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb_aggregation)

    print(f"\n--- SWB-DM extended check under '{attack}' ({num_rounds} rounds) ---")
    trajectory = []
    for r in range(1, num_rounds + 1):
        sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
        states, ids = [], []
        for cid in sampled:
            att = attack if cid in byz else None
            worker.load_state_dict(global_model.state_dict())
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
            states.append({k: v.detach().clone() for k, v in st.items()})
            ids.append(cid)
        global_model.load_state_dict(cache.step(ids, states))

        if r % checkpoint_every == 0 or r == 1:
            acc = evaluate(global_model, test_loader)
            trajectory.append((r, acc))
            print(f"  round {r:2d}/{num_rounds} -> acc={acc:.2f}%")

    round10_acc = next((a for rr, a in trajectory if rr >= 10), trajectory[-1][1])
    final_acc = trajectory[-1][1]
    print(f"\nSWB-DM under '{attack}': round~10={round10_acc:.2f}% -> round{num_rounds}={final_acc:.2f}%")
    if final_acc > round10_acc + 5:
        print("-> RECOVERING under attack too: consistent with cache warm-up, "
              "not attack-specific suppression.")
    elif final_acc <= round10_acc + 2:
        print("-> STILL FLAT under sustained attack: this suggests the attack, "
              "not just warm-up, is genuinely suppressing SWB-DM here -- worth "
              "investigating further before reporting the 10-round number as representative.")
    else:
        print("-> Partial recovery: inconclusive, consider extending further.")
    return trajectory


if __name__ == "__main__":
    np.random.seed(42)
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    counts = [len(l.dataset) for l in client_loaders]

    table = {m: [] for m in METHODS}
    print(f"Running: Naive Attacks @ {int(CORRUPTION*100)}% corruption\n")
    for method in METHODS:
        for attack in ATTACKS:
            debug = (method == "FLTrust")  # print trust-score debug only for FLTrust
            acc = run(method, attack, client_loaders, test_loader, root_loader, counts, debug_trust=debug)
            table[method].append(acc)
            print(f"{method:<9} | {attack:<11} | {acc:.2f}%", flush=True)

    x = np.arange(len(ATTACKS)); w = 0.11
    plt.figure(figsize=(12, 6))
    for i, m in enumerate(METHODS):
        plt.bar(x + i * w, table[m], width=w, label=m)
    plt.xticks(x + w * 3, ATTACKS, fontsize=12)
    plt.ylabel("Test accuracy (%)", fontsize=12)
    plt.title(f"Naive Attacks @ {int(CORRUPTION*100)}% Corruption (CIFAR-10, 10 Rounds)", fontsize=14)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("naive_attacks.png", dpi=150)
    print("\nSaved naive_attacks.png")

    # Extended SWB-DM check under one attack, to separate warm-up from
    # genuine attack suppression before trusting the 10-round comparison above.
    run_extended_swb_dm("sign_flip", client_loaders, test_loader, num_rounds=30)
