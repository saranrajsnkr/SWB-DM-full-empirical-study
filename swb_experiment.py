# SWB Aggregator Validation
import torch
import numpy as np
import copy
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import coordinate_wise_median, krum_aggregation, swb_aggregation
from cache import DelayedMomentumCache

def run_experiment():
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Loading data and partitioning among 20 clients...")
    client_loaders, test_loader = get_cifar10_loaders(num_clients=20, alpha=0.5)
    
    methods = {
        "Median": coordinate_wise_median,
        "Krum": lambda states: krum_aggregation(states, num_byzantine=2),
        "SWB": swb_aggregation,
        "SWB-DM": lambda states: swb_aggregation(states) # Wrapped in cache below
    }
    
    for method_name, agg_fn in methods.items():
        torch.manual_seed(42)
        np.random.seed(42)
        global_model = SimpleCNN().to(device)
        
        cache = None
        if method_name == "SWB-DM":
            cache = DelayedMomentumCache(global_model, 20, swb_aggregation)
            
        print(f"\n--- Testing {method_name} (10 Rounds, No Attacks) ---")
        for r in range(1, 11):
            sampled_ids = np.random.choice(20, 10, replace=False)
            sampled_states = []
            
            for cid in sampled_ids:
                local_model = copy.deepcopy(global_model)
                state = train_client(local_model, client_loaders[cid], epochs=2, lr=0.01)
                sampled_states.append(state)
                
            if cache is not None:
                agg_state = cache.step(sampled_ids, sampled_states)
                global_model.load_state_dict(agg_state)
            else:
                global_model.load_state_dict(agg_fn(sampled_states))
                
            if r % 2 == 0 or r == 1:
                acc = evaluate(global_model, test_loader)
                print(f"Round [{r}/10] - Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    run_experiment()