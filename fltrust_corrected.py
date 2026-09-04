import torch


def fltrust_aggregation(client_states, server_update, global_state):
    """
    FLTrust (Cao et al. 2021): weights client updates by ReLU-clipped cosine
    similarity to a clean server update, AND rescales each client update's
    norm to match the server update's norm before aggregating.

    The norm-clipping step is essential, not optional: without it, a
    Byzantine client whose update points in roughly the correct direction
    (passing the cosine-similarity filter) can still submit an update of
    unbounded magnitude and dominate the weighted sum. Norm-clipping bounds
    each client's contribution to at most the server's own update scale,
    regardless of how large the client's raw update actually was.
    """
    keys = client_states[0].keys()
    client_updates = torch.stack(
        [torch.cat([(s[k].float() - global_state[k].float()).view(-1) for k in keys])
         for s in client_states]
    )
    server_flat = torch.cat(
        [(server_update[k].float() - global_state[k].float()).view(-1) for k in keys]
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        client_updates, server_flat.unsqueeze(0), dim=1
    )
    trust_scores = torch.relu(cos_sim)  # drop negative similarities

    # --- Norm-clipping (the previously-missing step) ---
    server_norm = torch.norm(server_flat)
    client_norms = torch.norm(client_updates, dim=1, keepdim=True) + 1e-8
    scaled_updates = client_updates * (server_norm / client_norms)
    # ----------------------------------------------------

    if trust_scores.sum() < 1e-8:
        # No client passed the trust filter at all; fall back to a uniform
        # (still norm-clipped) average rather than an unweighted raw mean.
        weights = torch.ones(len(client_states), device=client_updates.device) / len(client_states)
    else:
        weights = trust_scores / trust_scores.sum()

    weighted_update = torch.sum(scaled_updates * weights.view(-1, 1), dim=0)

    agg = {}
    idx = 0
    for k in keys:
        numel = global_state[k].numel()
        agg[k] = global_state[k].float() + weighted_update[idx:idx + numel].view(global_state[k].shape)
        idx += numel
    return agg
