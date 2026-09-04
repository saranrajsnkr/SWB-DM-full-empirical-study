# Aggregation methods: FedAvg, Median, Krum, and Sliced Wasserstein Barycenter (SWB)
import torch
import numpy as np

def fed_avg(global_model, client_states, client_sample_counts):
    total_samples = sum(client_sample_counts)
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float)
        for state, num_samples in zip(client_states, client_sample_counts):
            global_dict[key] += state[key] * (num_samples / total_samples)
    global_model.load_state_dict(global_dict)
    return global_model

def coordinate_wise_median(client_states):
    aggregated_dict = {}
    keys = client_states[0].keys()
    for key in keys:
        stacked = torch.stack([state[key].float() for state in client_states], dim=0)
        aggregated_dict[key] = torch.median(stacked, dim=0).values
    return aggregated_dict

def krum_aggregation(client_states, num_byzantine=2):
    n = len(client_states)
    keys = client_states[0].keys()
    flat_vectors = [torch.cat([state[k].float().view(-1) for k in keys]) for state in client_states]
    flat_vectors = torch.stack(flat_vectors)
    
    scores = []
    for i in range(n):
        distances = torch.norm(flat_vectors - flat_vectors[i], dim=1)
        sorted_dists, _ = torch.sort(distances)
        score = torch.sum(sorted_dists[1 : n - num_byzantine - 1])
        scores.append(score.item())
        
    best_idx = np.argmin(scores)
    return client_states[best_idx]

def swb_aggregation(client_states, num_passes=2, chunk=512, trim_ratio=0.2):
    """
    Sliced Wasserstein Barycenter (SWB) Aggregation.
    Projects client updates onto random orthogonal slices. In 1D, the W2
    barycenter is the mean of sorted values. We apply a trimmed mean for
    Byzantine robustness, then rotate back. Coordinate identity is restored
    using the Wasserstein medoid as a gauge.
    """
    keys = list(client_states[0].keys())
    N = len(client_states)
    flats = [torch.cat([s[k].float().view(-1) for k in keys]) for s in client_states]
    D = flats[0].numel()
    trim_k = int(N * trim_ratio)
    out = torch.zeros(D, device=flats[0].device)
    
    for _ in range(num_passes):
        for i in range(0, D, chunk):
            x = torch.stack([f[i:i + chunk] for f in flats])  # [N, C]
            C = x.shape[1]
            if C > 1:
                q, _ = torch.linalg.qr(torch.randn(C, C, device=x.device))
                y = x @ q  # Random projection (slice)
            else:
                y = x
                
            srt, perm = torch.sort(y, dim=1)  # 1D OT closed form
            
            # Robust W2 barycenter (trimmed mean of sorted values)
            b = torch.mean(torch.sort(srt, dim=0).values[trim_k:N - trim_k], dim=0)
            
            if C > 1:
                # Wasserstein medoid for gauge fixing
                m = int(torch.argmin(torch.norm(srt - b, dim=1)))
                o = b[torch.argsort(perm[m])]
                out[i:i + C] += o @ q.T
            else:
                out[i:i + C] += b
                
    out /= num_passes
    
    agg = {}
    idx = 0
    for k in keys:
        ne = client_states[0][k].numel()
        agg[k] = out[idx:idx + ne].view(client_states[0][k].shape)
        idx += ne
    return agg

def bulyan_aggregation(client_states, num_byzantine=2):
    """
    Bulyan: 1) Select (n - 2f) updates using Krum's multi-selection rule.
            2) Apply coordinate-wise trimmed mean on the selected subset.
    """
    n = len(client_states)
    keys = client_states[0].keys()
    flat_vectors = torch.stack([torch.cat([s[k].float().view(-1) for k in keys]) for s in client_states])
    
    num_to_select = max(1, n - 2 * num_byzantine)
    selected_indices = []
    remaining = list(range(n))
    
    for _ in range(num_to_select):
        scores = []
        for i in remaining:
            dists = sorted([torch.norm(flat_vectors[i] - flat_vectors[j]).item() 
                           for j in remaining if i != j])
            scores.append(sum(dists[:max(0, len(remaining) - num_byzantine - 2)]))
        best_local = np.argmin(scores)
        best_global = remaining[best_local]
        selected_indices.append(best_global)
        remaining.remove(best_global)
        
    selected_states = [client_states[i] for i in selected_indices]
    trim_k = min(num_byzantine, (len(selected_states) - 1) // 2)
    
    agg = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in selected_states], dim=0)
        if trim_k > 0:
            srt, _ = torch.sort(stacked, dim=0)
            agg[k] = torch.mean(srt[trim_k : len(selected_states) - trim_k], dim=0)
        else:
            agg[k] = torch.mean(stacked, dim=0)
    return agg

def fltrust_aggregation(client_states, server_update, global_state):
    """
    FLTrust: Weights client updates by cosine similarity to a clean server update.
    """
    keys = client_states[0].keys()
    client_updates = torch.stack([torch.cat([(s[k].float() - global_state[k].float()).view(-1) for k in keys]) for s in client_states])
    server_flat = torch.cat([(server_update[k].float() - global_state[k].float()).view(-1) for k in keys])
    
    cos_sim = torch.nn.functional.cosine_similarity(client_updates, server_flat.unsqueeze(0), dim=1)
    trust_scores = torch.relu(cos_sim) # Drop negative similarities
    
    if trust_scores.sum() < 1e-8:
        weights = torch.ones(len(client_states)) / len(client_states)
    else:
        weights = trust_scores / trust_scores.sum()
        
    agg = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in client_states], dim=0)
        agg[k] = torch.sum(stacked * weights.view(-1, *([1]*(stacked.dim()-1))), dim=0)
    return agg