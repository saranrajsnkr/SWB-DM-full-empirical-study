# experiment_attacks_extended.py: 30-round attack recovery check
import torch
import numpy as np
import copy
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import coordinate_wise_median, swb_aggregation
from cache import DelayedMomentumCache

NUM_ROUNDS = 30
CORRUPTION = 0.2
NUM_CLIENTS = 20
ATTACK = "sign_flip"

def run_extended(method_name, agg_fn, is_cache, client_loaders, test_loader):
    torch.manual_seed(42); np.random.seed(42)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))
    global_model = SimpleCNN().to(device)
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, agg_fn) if is_cache else None
    
    print(f"\n--- {method_name} ({NUM_ROUNDS} rounds, {ATTACK} @ {int(CORRUPTION*100)}%) ---")
    for r in range(1, NUM_ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, 10, replace=False)
        states = []
        for cid in sampled:
            worker = SimpleCNN().to(device)
            worker.load_state_dict(global_model.state_dict())
            att = ATTACK if cid in byz else None
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
            states.append(st)
            
        if is_cache:
            global_model.load_state_dict(cache.step(list(sampled), states))
        else:
            global_model.load_state_dict(agg_fn(states))
            
        if r % 5 == 0 or r == 1:
            print(f"Round [{r:2d}/{NUM_ROUNDS}] - Acc: {evaluate(global_model, test_loader):.2f}%")

if __name__ == "__main__":
    print("Loading data...")
    client_loaders, test_loader, _ = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    
    # Run Median (non-cache baseline)
    run_extended("Median", coordinate_wise_median, is_cache=False, 
                 client_loaders=client_loaders, test_loader=test_loader)
                 
    # Run SWB-DM (cache method)
    run_extended("SWB-DM", swb_aggregation, is_cache=True, 
                 client_loaders=client_loaders, test_loader=test_loader)