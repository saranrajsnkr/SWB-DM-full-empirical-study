# debug_fltrust.py: Proves FLTrust is actually ignoring Byzantine clients
import torch
import numpy as np
from common import SimpleCNN, get_cifar10_loaders, train_client, device

def fltrust_debug(client_states, server_update, global_state):
    """Returns weights, cosine similarities, and raw trust scores for inspection."""
    keys = client_states[0].keys()
    client_updates = torch.stack([torch.cat([(s[k].float() - global_state[k].float()).view(-1) for k in keys]) for s in client_states])
    server_flat = torch.cat([(server_update[k].float() - global_state[k].float()).view(-1) for k in keys])
    
    cos_sim = torch.nn.functional.cosine_similarity(client_updates, server_flat.unsqueeze(0), dim=1)
    trust_scores = torch.relu(cos_sim)  # The ReLU filter
    
    server_norm = torch.norm(server_flat)
    client_norms = torch.norm(client_updates, dim=1, keepdim=True) + 1e-8
    scaled_updates = client_updates * (server_norm / client_norms)
    
    if trust_scores.sum() < 1e-8:
        weights = torch.ones(len(client_states)) / len(client_states)
    else:
        weights = trust_scores / trust_scores.sum()
        
    return weights, cos_sim, trust_scores

if __name__ == "__main__":
    np.random.seed(42); torch.manual_seed(42)
    client_loaders, _, root_loader = get_cifar10_loaders(num_clients=20, alpha=0.5)
    global_model = SimpleCNN().to(device)
    server_model = SimpleCNN().to(device)

    # Simulate Round 1 with 20% corruption (Clients 0-3 are Byzantine)
    sampled = np.random.choice(20, 10, replace=False)
    byz_ids = set(range(4)) 
    states = []
    
    print("Training 10 clients (4 Byzantine with sign_flip, 6 Benign)...")
    for cid in sampled:
        worker = SimpleCNN().to(device)
        worker.load_state_dict(global_model.state_dict())
        att = "sign_flip" if cid in byz_ids else None
        st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01, attack_type=att)
        states.append(st)

    server_model.load_state_dict(global_model.state_dict())
    srv = train_client(server_model, root_loader, epochs=1, lr=0.01)

    weights, cos_sim, trust = fltrust_debug(states, srv, global_model.state_dict())

    print(f"\n{'Client ID':>9} | {'Byzantine?':>10} | {'Cosine Sim':>10} | {'Trust Score':>11} | {'Final Weight':>12}")
    print("-" * 70)
    for i, cid in enumerate(sampled):
        is_byz = "YES" if cid in byz_ids else "no"
        print(f"{cid:>9} | {is_byz:>10} | {cos_sim[i].item():>10.4f} | {trust[i].item():>11.4f} | {weights[i].item():>12.4f}")