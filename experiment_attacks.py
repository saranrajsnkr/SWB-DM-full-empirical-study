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

def run(method, attack, client_loaders, test_loader, root_loader, counts, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))        # clients 0-3 are Byzantine
    n_sample = 11 if method == "Bulyan" else 10            # Bulyan precondition n >= 4f+3
    f_assumed = int(n_sample * CORRUPTION)  # Dynamic calculation (e.g. 10*0.2=2, 11*0.2=2)
    global_model = SimpleCNN().to(device)
    worker = SimpleCNN().to(device)                        # hoisted worker (speed)
    server_model = SimpleCNN().to(device) if method == "FLTrust" else None
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb_aggregation) if method == "SWB-DM" else None

    for r in range(1, NUM_ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
        states, s_counts, ids = [], [], []
        for cid in sampled:
            att = attack if cid in byz else None
            worker.load_state_dict(global_model.state_dict())
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
            states.append({k: v.detach().clone() for k, v in st.items()})   # clone = essential
            s_counts.append(counts[cid]); ids.append(cid)

        if cache is not None:
            global_model.load_state_dict(cache.step(ids, states))
        elif method == "FedAvg":
            global_model = fed_avg(global_model, states, s_counts)
        elif method == "Median":
            global_model.load_state_dict(coordinate_wise_median(states))
        elif method == "Krum":
            global_model.load_state_dict(krum_aggregation(states, num_byzantine=f_assumed))
        elif method == "Bulyan":
            global_model.load_state_dict(bulyan_aggregation(states, num_byzantine=f_assumed))
        elif method == "FLTrust":
            server_model.load_state_dict(global_model.state_dict())
            srv = train_client(server_model, root_loader, epochs=1, lr=0.01)
            global_model.load_state_dict(fltrust_aggregation(states, srv, global_model.state_dict()))
        elif method == "SWB":
            global_model.load_state_dict(swb_aggregation(states))
    return evaluate(global_model, test_loader)

if __name__ == "__main__":
    np.random.seed(42)
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    counts = [len(l.dataset) for l in client_loaders]

    table = {m: [] for m in METHODS}
    print(f"Running: Naive Attacks @ {int(CORRUPTION*100)}% corruption\n")
    for method in METHODS:
        for attack in ATTACKS:
            acc = run(method, attack, client_loaders, test_loader, root_loader, counts)
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