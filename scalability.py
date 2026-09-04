#500-client scalability (clean): wall-clock, communication, warm-up at scale
import csv, time
import numpy as np
import torch
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (fed_avg, coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache

NUM_CLIENTS, SAMPLE_RATE = 500, 0.1
MAIN_ROUNDS, EXT_ROUNDS, CHECKPOINT_EVERY = 10, 40, 5
METHODS = ["FedAvg", "Median", "Krum", "Bulyan", "FLTrust", "SWB", "DelayedMomentum", "SWB-DM"]
CACHE_METHODS = {"DelayedMomentum", "SWB-DM"}

def sync():
    if device.type == "cuda": torch.cuda.synchronize()

class CPUPoolCache:
    """Full-pool cache on CPU (500 GPU states would OOM)."""
    def __init__(self, model, n):
        self.n = n
        self.buffer = {i: {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items()} for i in range(n)}
    def update(self, ids, states):
        for cid, s in zip(ids, states):
            self.buffer[cid] = {k: v.detach().cpu().float() for k, v in s.items()}
    def pool(self):
        return [self.buffer[i] for i in range(self.n)]

def median_cpu(pool):
    keys = list(pool[0].keys())
    return {k: torch.median(torch.stack([s[k] for s in pool]), dim=0).values for k in keys}

def swb_cpu_chunked(pool, num_passes=2, chunk=2048, trim_ratio=0.2):
    keys = list(pool[0].keys()); N = len(pool)
    flats = [torch.cat([s[k].view(-1) for k in keys]) for s in pool]
    D = flats[0].numel()
    trim_k = min(int(N * trim_ratio), (N - 1) // 2)
    out = torch.zeros(D)
    for _ in range(num_passes):
        for i in range(0, D, chunk):
            x = torch.stack([f[i:i + chunk] for f in flats]).to(device)
            C = x.shape[1]
            if C > 1:
                q, _ = torch.linalg.qr(torch.randn(C, C, device=device))
                y = x @ q
            else:
                y = x
            srt, perm = torch.sort(y, dim=1)
            b = torch.mean(torch.sort(srt, dim=0).values[trim_k:N - trim_k], dim=0)
            if C > 1:
                m = int(torch.argmin(torch.norm(srt - b, dim=1)))
                out[i:i + C] += (b[torch.argsort(perm[m])] @ q.T).cpu()
            else:
                out[i:i + C] += b.cpu()
    out /= num_passes
    agg, idx = {}, 0
    for k in keys:
        ne = pool[0][k].numel()
        agg[k] = out[idx:idx + ne].view(pool[0][k].shape); idx += ne
    return agg

def run_method(method, rounds, log_traj=False):
    torch.manual_seed(42); np.random.seed(42)
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    num_sampled = int(NUM_CLIENTS * SAMPLE_RATE)
    model, worker = SimpleCNN().to(device), SimpleCNN().to(device)
    server = SimpleCNN().to(device) if method == "FLTrust" else None
    cache = CPUPoolCache(model, NUM_CLIENTS) if method in CACHE_METHODS else None
    inner = median_cpu if method == "DelayedMomentum" else swb_cpu_chunked
    train_t, agg_t, traj = [], [], []
    t_start = time.time()
    for r in range(1, rounds + 1):
        sampled = np.random.choice(NUM_CLIENTS, num_sampled, replace=False)
        current = model.state_dict()
        sync(); t0 = time.time()
        states, s_counts = [], []
        for cid in sampled:
            worker.load_state_dict(current)
            st = train_client(worker, client_loaders[cid], epochs=2, lr=0.01)
            states.append({k: v.detach().clone() for k, v in st.items()})
            s_counts.append(len(client_loaders[cid].dataset))
        sync(); train_t.append(time.time() - t0)
        sync(); t0 = time.time()
        if cache is not None:
            cache.update(list(sampled), states)
            model.load_state_dict(inner(cache.pool()))
        elif method == "FedAvg":  model = fed_avg(model, states, s_counts)
        elif method == "Median":  model.load_state_dict(coordinate_wise_median(states))
        elif method == "Krum":    model.load_state_dict(krum_aggregation(states, num_byzantine=0))
        elif method == "Bulyan":  model.load_state_dict(bulyan_aggregation(states, num_byzantine=0))
        elif method == "FLTrust":
            server.load_state_dict(current)
            srv = train_client(server, root_loader, epochs=1, lr=0.01)
            model.load_state_dict(fltrust_aggregation(states, srv, current))
        elif method == "SWB":     model.load_state_dict(swb_aggregation(states, trim_ratio=0.2))
        sync(); agg_t.append(time.time() - t0)
        if log_traj and (r % CHECKPOINT_EVERY == 0 or r == 1):
            acc = evaluate(model, test_loader)
            traj.append((r, acc))
            print(f"  round {r:3d}/{rounds} -> acc={acc:.2f}%", flush=True)
    final_acc = evaluate(model, test_loader)
    return dict(method=method, rounds=rounds,
                total_wall_sec=round(time.time() - t_start, 1),
                mean_train_sec_round=round(sum(train_t) / rounds, 2),
                mean_agg_sec_round=round(sum(agg_t) / rounds, 2),
                agg_pool_size=NUM_CLIENTS if method in CACHE_METHODS else num_sampled,
                final_acc=round(final_acc, 2)), traj

if __name__ == "__main__":
    np.random.seed(123); torch.manual_seed(42)
    client_loaders, test_loader, root_loader = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    param_count = sum(p.numel() for p in SimpleCNN().parameters())
    comm_mb = int(NUM_CLIENTS * SAMPLE_RATE) * param_count * 4 / 1e6
    print(f"comm/round (transmission) = {comm_mb:.1f} MB for ALL methods", flush=True)

    rows = []
    for method in METHODS:
        print(f"\n>>> {method} | 500 clients | {MAIN_ROUNDS} rounds", flush=True)
        row, _ = run_method(method, MAIN_ROUNDS)
        row["comm_mb_round_transmission"] = round(comm_mb, 1)
        rows.append(row); print(row, flush=True)
    with open("scalability_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("Saved scalability_results.csv", flush=True)

    ext_rows, trajs = [], {}
    for method in ["DelayedMomentum", "SWB-DM"]:
        print(f"\n>>> {method} | 500 clients | {EXT_ROUNDS} rounds (extended)", flush=True)
        row, traj = run_method(method, EXT_ROUNDS, log_traj=True)
        ext_rows.append(row); trajs[method] = traj
    with open("scalability_extended_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ext_rows[0].keys())); w.writeheader(); w.writerows(ext_rows)
    print("Saved scalability_extended_results.csv", flush=True)
    for m, traj in trajs.items():
        first, last = traj[0][1], traj[-1][1]
        print(f"{m}: r1={first:.2f}% -> r{EXT_ROUNDS}={last:.2f}%")
        if last > 15.0 and last > first + 3:
            print("  -> RECOVERING: cache warm-up at scale, mechanism scales correctly")
        elif last <= 12.0:
            print("  -> STILL NEAR CHANCE: investigate, not just warm-up")
        else:
            print("  -> partial improvement, inconclusive")
    print("SCALABILITY COMPLETE", flush=True)