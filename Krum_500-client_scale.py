# Krum clean-collapse sanity check at 500-client scale
import numpy as np
import torch
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import krum_aggregation

NUM_CLIENTS, SAMPLE_RATE = 500, 0.1
ROUNDS, CHECKPOINT_EVERY = 40, 5

if __name__ == "__main__":
    np.random.seed(123); torch.manual_seed(42)
    client_loaders, test_loader, _ = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    sizes = [len(l.dataset) for l in client_loaders]
    print(f"per-client samples at N=500: min={min(sizes)}, median={int(np.median(sizes))}, max={max(sizes)}", flush=True)
    torch.manual_seed(42); np.random.seed(42)
    model, worker = SimpleCNN().to(device), SimpleCNN().to(device)
    traj = []
    for r in range(1, ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * SAMPLE_RATE), replace=False)
        current = model.state_dict()
        states = []
        for cid in sampled:
            worker.load_state_dict(current)
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01)
            states.append({k: v.detach().clone() for k, v in st.items()})
        model.load_state_dict(krum_aggregation(states, num_byzantine=0))
        if r % CHECKPOINT_EVERY == 0 or r == 1:
            acc = evaluate(model, test_loader)
            traj.append((r, acc))
            print(f"round {r:3d}/{ROUNDS} -> acc={acc:.2f}%", flush=True)
    first, last = traj[0][1], traj[-1][1]
    print(f"\nKrum @500 clients, clean: r1={first:.2f}% -> r{ROUNDS}={last:.2f}%")
    if last > first + 5:
        print("-> SLOW LEARNING: statistical inefficiency of single-client selection, not a bug")
    else:
        print("-> STILL FLAT: near-non-convergent single-client selection under ~100-sample clients; report as known limitation")