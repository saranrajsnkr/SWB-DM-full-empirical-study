# Extended validation: does SWB-DM recover with more rounds, matching the
# warm-up pattern already observed for RPM-DM's cache mechanism?
import torch
import numpy as np
import copy
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import coordinate_wise_median, krum_aggregation, swb_aggregation
from cache import DelayedMomentumCache

NUM_ROUNDS = 30          # up from 10
CHECKPOINT_EVERY = 2
NUM_CLIENTS = 20
NUM_SAMPLED = 10


def run_method(method_name, agg_fn, is_cache, client_loaders, test_loader):
    torch.manual_seed(42)
    np.random.seed(42)
    global_model = SimpleCNN().to(device)
    cache = DelayedMomentumCache(global_model, NUM_CLIENTS, agg_fn) if is_cache else None

    print(f"\n--- {method_name} ({NUM_ROUNDS} rounds, no attack) ---")
    trajectory = []
    for r in range(1, NUM_ROUNDS + 1):
        sampled_ids = np.random.choice(NUM_CLIENTS, NUM_SAMPLED, replace=False)
        sampled_states = []
        for cid in sampled_ids:
            local_model = copy.deepcopy(global_model)
            state = train_client(local_model, client_loaders[cid], epochs=2, lr=0.01)
            sampled_states.append(state)

        if is_cache:
            agg_state = cache.step(sampled_ids, sampled_states)
            global_model.load_state_dict(agg_state)
        else:
            global_model.load_state_dict(agg_fn(sampled_states))

        if r % CHECKPOINT_EVERY == 0 or r == 1:
            acc = evaluate(global_model, test_loader)
            trajectory.append((r, acc))
            print(f"Round [{r:2d}/{NUM_ROUNDS}] - Test Accuracy: {acc:.2f}%")

    return trajectory


if __name__ == "__main__":
    print("Loading data and partitioning among 20 clients...")
    client_loaders, test_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)

    results = {}
    results["Median"] = run_method(
        "Median", coordinate_wise_median, is_cache=False,
        client_loaders=client_loaders, test_loader=test_loader)
    results["Krum"] = run_method(
        "Krum", lambda states: krum_aggregation(states, num_byzantine=2), is_cache=False,
        client_loaders=client_loaders, test_loader=test_loader)
    results["SWB"] = run_method(
        "SWB", swb_aggregation, is_cache=False,
        client_loaders=client_loaders, test_loader=test_loader)
    results["SWB-DM"] = run_method(
        "SWB-DM", swb_aggregation, is_cache=True,
        client_loaders=client_loaders, test_loader=test_loader)

    print("\n" + "=" * 60)
    print(f"SUMMARY -- {NUM_ROUNDS}-round extended comparison (no attack)")
    print("=" * 60)
    for name, traj in results.items():
        first_r, first_acc = traj[0]
        last_r, last_acc = traj[-1]
        ten_round_acc = next((a for r, a in traj if r >= 10), last_acc)
        print(f"{name:<10} round{first_r:>3}={first_acc:5.2f}%  "
              f"round~10={ten_round_acc:5.2f}%  round{last_r:>3}={last_acc:5.2f}%")

    # Recovery check specific to SWB-DM vs its non-cached counterpart
    swb_dm_10 = next((a for r, a in results["SWB-DM"] if r >= 10), None)
    swb_dm_final = results["SWB-DM"][-1][1]
    swb_final = results["SWB"][-1][1]
    print(f"\nSWB-DM: round~10={swb_dm_10:.2f}% -> round{NUM_ROUNDS}={swb_dm_final:.2f}%")
    if swb_dm_final > swb_dm_10 + 5 and swb_dm_final >= swb_final - 5:
        print("-> RECOVERING: consistent with cache warm-up (same pattern as RPM-DM), "
              "not a regression in the SWB aggregator itself.")
    elif swb_dm_final <= swb_dm_10 + 2:
        print("-> STILL FLAT after extension: investigate the medoid gauge-fixing step "
              "and cache interaction further before trusting the 10-round result as final.")
    else:
        print("-> Partial improvement: consider extending further or reporting the "
              "trajectory directly rather than a single endpoint.")
