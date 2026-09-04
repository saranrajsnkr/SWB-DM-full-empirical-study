# Full Baseline & SWB-DM Validation
import torch
import numpy as np
import copy
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (coordinate_wise_median, krum_aggregation, 
                         bulyan_aggregation, fltrust_aggregation, swb_aggregation)
from cache import DelayedMomentumCache

def run_experiment():
    torch.manual_seed(42); np.random.seed(42)
    print("Loading data and partitioning among 20 clients...")
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=20, alpha=0.5)
    
    methods = {
        "Median": coordinate_wise_median,
        "Krum": lambda s: krum_aggregation(s, num_byzantine=2),
        "Bulyan": lambda s: bulyan_aggregation(s, num_byzantine=2),
        "SWB": swb_aggregation,
    }
    
    for name, agg_fn in methods.items():
        torch.manual_seed(42); np.random.seed(42)
        global_model = SimpleCNN().to(device)
        print(f"\n--- {name} (10 Rounds, Clean) ---")
        for r in range(1, 11):
            sampled_ids = np.random.choice(20, 10, replace=False)
            states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01) for c in sampled_ids]
            global_model.load_state_dict(agg_fn(states))
            if r % 2 == 0 or r == 1: print(f"Round [{r}/10] - Acc: {evaluate(global_model, test_loader):.2f}%")

    # FLTrust requires server training
    torch.manual_seed(42); np.random.seed(42)
    global_model = SimpleCNN().to(device)
    server_model = SimpleCNN().to(device)
    print(f"\n--- FLTrust (10 Rounds, Clean) ---")
    for r in range(1, 11):
        server_model.load_state_dict(global_model.state_dict())
        srv_state = train_client(server_model, root_loader, epochs=1, lr=0.01)
        sampled_ids = np.random.choice(20, 10, replace=False)
        states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01) for c in sampled_ids]
        global_model.load_state_dict(fltrust_aggregation(states, srv_state, global_model.state_dict()))
        if r % 2 == 0 or r == 1: print(f"Round [{r}/10] - Acc: {evaluate(global_model, test_loader):.2f}%")

    # SWB-DM
    torch.manual_seed(42); np.random.seed(42)
    global_model = SimpleCNN().to(device)
    cache = DelayedMomentumCache(global_model, 20, swb_aggregation)
    print(f"\n--- SWB-DM (10 Rounds, Clean) ---")
    for r in range(1, 11):
        sampled_ids = np.random.choice(20, 10, replace=False)
        states = [train_client(copy.deepcopy(global_model), client_loaders[c], epochs=2, lr=0.01) for c in sampled_ids]
        global_model.load_state_dict(cache.step(sampled_ids, states))
        if r % 2 == 0 or r == 1: print(f"Round [{r}/10] - Acc: {evaluate(global_model, test_loader):.2f}%")

if __name__ == "__main__":
    run_experiment()