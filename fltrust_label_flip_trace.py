# fltrust_label_flip_trace.py
# Traces FLTrust's Byzantine vs. honest trust weights EVERY round (not just
# round 1 and round 10) under label_flip specifically,
# debug output showed label_flip is the one attack where Byzantine clients
# receive non-trivial trust weight. This checks whether that gap persists,
# widens, or fades as training progresses.
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SimpleCNN, get_cifar10_loaders, train_client, device

NUM_CLIENTS = 20
NUM_ROUNDS = 10
CORRUPTION = 0.2
ATTACK = "label_flip"


def compute_trust(states, ids, byz, server_update, global_state):
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

    server_norm = torch.norm(server_flat)
    client_norms = torch.norm(client_updates, dim=1, keepdim=True) + 1e-8
    scaled_updates = client_updates * (server_norm / client_norms)
    weighted_update = torch.sum(scaled_updates * weights.view(-1, 1), dim=0)

    agg = {}
    idx = 0
    for k in keys:
        numel = global_state[k].numel()
        agg[k] = global_state[k].float() + weighted_update[idx:idx + numel].view(global_state[k].shape)
        idx += numel

    byz_w = [weights[i].item() for i, cid in enumerate(ids) if cid in byz]
    hon_w = [weights[i].item() for i, cid in enumerate(ids) if cid not in byz]
    byz_cos = [cos_sim[i].item() for i, cid in enumerate(ids) if cid in byz]
    hon_cos = [cos_sim[i].item() for i, cid in enumerate(ids) if cid not in byz]
    return agg, byz_w, hon_w, byz_cos, hon_cos


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    byz = set(range(int(NUM_CLIENTS * CORRUPTION)))

    global_model = SimpleCNN().to(device)
    worker = SimpleCNN().to(device)
    server_model = SimpleCNN().to(device)

    byz_weight_traj, hon_weight_traj = [], []
    byz_cos_traj, hon_cos_traj = [], []

    print(f"--- FLTrust full trust trace under '{ATTACK}' @ {int(CORRUPTION*100)}% corruption ---")
    for r in range(1, NUM_ROUNDS + 1):
        sampled = np.random.choice(NUM_CLIENTS, 10, replace=False)
        states, ids = [], []
        for cid in sampled:
            att = ATTACK if cid in byz else None
            worker.load_state_dict(global_model.state_dict())
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
            states.append({k: v.detach().clone() for k, v in st.items()})
            ids.append(cid)

        server_model.load_state_dict(global_model.state_dict())
        srv = train_client(server_model, root_loader, epochs=1, lr=0.01)
        current_global = global_model.state_dict()

        agg, byz_w, hon_w, byz_cos, hon_cos = compute_trust(states, ids, byz, srv, current_global)
        global_model.load_state_dict(agg)

        byz_mean, hon_mean = np.mean(byz_w), np.mean(hon_w)
        byz_weight_traj.append(byz_mean)
        hon_weight_traj.append(hon_mean)
        byz_cos_traj.append(np.mean(byz_cos))
        hon_cos_traj.append(np.mean(hon_cos))

        print(f"round {r:2d}: byz_weight={byz_mean:.4f} (n={len(byz_w)}) | hon_weight={hon_mean:.4f} "
              f"| byz_cos={np.mean(byz_cos):+.4f} | hon_cos={np.mean(hon_cos):+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rounds = list(range(1, NUM_ROUNDS + 1))

    axes[0].plot(rounds, byz_weight_traj, 'o-', color='red', label='Byzantine (mean)')
    axes[0].plot(rounds, hon_weight_traj, 's-', color='green', label='Honest (mean)')
    axes[0].set_xlabel('Round'); axes[0].set_ylabel('FLTrust weight')
    axes[0].set_title(f'Trust weight over time ({ATTACK})')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(rounds, byz_cos_traj, 'o-', color='red', label='Byzantine (mean)')
    axes[1].plot(rounds, hon_cos_traj, 's-', color='green', label='Honest (mean)')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('Round'); axes[1].set_ylabel('Cosine similarity to server update')
    axes[1].set_title(f'Cosine similarity over time ({ATTACK})')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("fltrust_label_flip_trace.png", dpi=150)
    print("\nSaved fltrust_label_flip_trace.png")

    print("\n--- Summary ---")
    print(f"Byzantine weight: round1={byz_weight_traj[0]:.4f} -> round{NUM_ROUNDS}={byz_weight_traj[-1]:.4f}")
    if byz_weight_traj[-1] < byz_weight_traj[0] * 0.5:
        print("-> FADING: Byzantine trust weight drops substantially as training "
              "progresses -- the honest signal likely strengthens/stabilizes "
              "relative to label-noise-corrupted gradients over time.")
    elif byz_weight_traj[-1] > byz_weight_traj[0] * 1.5:
        print("-> WORSENING: Byzantine trust weight grows over time -- worth "
              "investigating why label-flip becomes MORE cosine-aligned with "
              "training progress, this would be an important limitation to report.")
    else:
        print("-> STABLE: Byzantine weight under label_flip stays roughly constant "
              "throughout training -- a persistent partial vulnerability, not a "
              )
