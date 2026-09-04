# Full Baseline & SWB-DM Validation
# Fixes applied:
#   1. Bulyan now samples n=11 clients (was 10) to satisfy its own
#      precondition n >= 4f+3 = 4(2)+3 = 11 at f=2. At n=10 the aggregator
#      ran without crashing, but outside its provable safety margin --
#      exactly the kind of silent precondition gap that caused Krum's
#      catastrophic divergence in earlier diagnostics, so we fix it before
#      any attack experiments are layered on top.
#   2. FLTrust now uses the corrected aggregator with norm-clipping.
import torch
import numpy as np
import copy
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache

NUM_CLIENTS = 20
N_STANDARD = 10          # sample size for Median / Krum / SWB / SWB-DM / FLTrust
N_BULYAN = 11            # bumped up to satisfy Bulyan's n >= 4f+3 precondition at f=2
NUM_ROUNDS = 10


def run_experiment():
    torch.manual_seed(42)
    np.random.seed(42)
    print("Loading data and partitioning among 20 clients...")
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)

    methods = {
        "Median": (coordinate_wise_median, N_STANDARD),
        "Krum": (lambda s: krum_aggregation(s, num_byzantine=2), N_STANDARD),
        "Bulyan": (lambda s: bulyan_aggregation(s, num_byzantine=2), N_BULYAN),
        "SWB": (swb_aggregation, N_STANDARD),
    }

    for name, (agg_fn, n_sample) in methods.items():
        torch.manual_seed(42)
        np.random.seed(42)
        global_model = SimpleCNN().to(device)
        print(f"\n--- {name} (n={n_sample}, {NUM_ROUNDS} Rounds, Clean) ---")
        for r in range(1, NUM_ROUNDS + 1):
            sampled_ids = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
            states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01)
                     for c in sampled_ids]
            global_model.load_state_dict(agg_fn(states))
            if r % 2 == 0 or r == 1:
                print(f"Round [{r}/{NUM_ROUNDS}] - Acc: {evaluate(global_model, test_loader):.2f}%")

    # FLTrust requires server training
    torch.manual_seed(42)
    np.random.seed(42)
    global_model = SimpleCNN().to(device)
    server_model = SimpleCNN().to(device)
    print(f"\n--- FLTrust (n={N_STANDARD}, {NUM_ROUNDS} Rounds, Clean, norm-clipped) ---")
    for r in range(1, NUM_ROUNDS + 1):
        server_model.load_state_dict(global_model.state_dict())
        srv_state = train_client(server_model, root_loader, epochs=1, lr=0.01)
        sampled_ids = np.random.choice(NUM_CLIENTS, N_STANDARD, replace=False)
        states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01)
                 for c in sampled_ids]
        global_model.load_state_dict(fltrust_aggregation(states, srv_state, global_model.state_dict()))
        if r % 2 == 0 or r == 1:
            print(f"Round [{r}/{NUM_ROUNDS}] - Acc: {evaluate(global_model, test_loader):.2f}%")

    # SWB-DM
    torch.manual_seed(42)
    np.random.seed(42)
    global_model = SimpleCNN().to(device)
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb_aggregation)
    print(f"\n--- SWB-DM (n={N_STANDARD}, {NUM_ROUNDS} Rounds, Clean) ---")
    for r in range(1, NUM_ROUNDS + 1):
        sampled_ids = np.random.choice(NUM_CLIENTS, N_STANDARD, replace=False)
        states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01)
                 for c in sampled_ids]
        global_model.load_state_dict(cache.step(sampled_ids, states))
        if r % 2 == 0 or r == 1:
            print(f"Round [{r}/{NUM_ROUNDS}] - Acc: {evaluate(global_model, test_loader):.2f}%")


if __name__ == "__main__":
    run_experiment()
