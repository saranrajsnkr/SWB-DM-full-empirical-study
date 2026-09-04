#Full Experiment Grid (448 configs)
import csv, os, time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (fed_avg, coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache
from attacks import generate_adaptive_attacks

NUM_CLIENTS, NUM_ROUNDS = 20, 10
METHODS = ["FedAvg", "Median", "Krum", "Bulyan", "FLTrust", "SWB", "SWB-DM"]
ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "alie", "ipm"]
CORRUPTIONS = [0.0, 0.1, 0.2, 0.3]
PARTICIPATIONS = [0.5, 0.1]
SEEDS = [42, 7]
RESULTS_FILE = "grid_results.csv"
NUM_WORKERS = 6
_cache = {}

def _worker_init():
    global _cache
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(123) # Fixed partition seed
    cl, tl, rl = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    _cache.update(loaders=cl, test=tl, root=rl, counts=[len(l.dataset) for l in cl])

def assumed_f(n, corr): return max(0, int(n * corr))
def trim_for(corr):     return min(0.45, corr + 0.05)

def _run_one(cfg):
    method, attack, corr, part, seed = cfg
    try:
        loaders, test_loader = _cache["loaders"], _cache["test"]
        root_loader, counts = _cache["root"], _cache["counts"]
        torch.manual_seed(seed); np.random.seed(seed)
        
        byz = set(range(int(NUM_CLIENTS * corr))) if corr > 0 else set()
        n_sample = 11 if method == "Bulyan" else 10
        if part < 0.5: n_sample = max(1, int(NUM_CLIENTS * part))
        f = assumed_f(n_sample, corr)
        trim = trim_for(corr)
        swb = lambda s: swb_aggregation(s, trim_ratio=trim)
        
        global_model = SimpleCNN().to(device)
        worker = SimpleCNN().to(device)
        server_model = SimpleCNN().to(device) if method == "FLTrust" else None
        cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb) if method == "SWB-DM" else None

        for r in range(1, NUM_ROUNDS + 1):
            sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
            current = global_model.state_dict()
            mal_ids = [c for c in sampled if c in byz]
            ben_ids = [c for c in sampled if c not in byz]
            
            ben_states, ben_counts = [], []
            for cid in ben_ids:
                worker.load_state_dict(current)
                st = train_client(worker, loaders[cid], epochs=2, lr=0.01)
                ben_states.append({k: v.detach().clone() for k, v in st.items()})
                ben_counts.append(counts[cid])
                
            mal_states, mal_counts = [], []
            if mal_ids and attack != "none":
                if attack in ("alie", "ipm") and ben_states:
                    bu = [{k: s[k].float() - current[k].float() for k in current} for s in ben_states]
                    mu = generate_adaptive_attacks(bu, attack, len(mal_ids), num_sampled_total=len(sampled))
                    mal_states = [{k: current[k].float() + u[k] for k in current} for u in mu]
                else:
                    for cid in mal_ids:
                        worker.load_state_dict(current)
                        st = train_client(worker, loaders[cid], epochs=2, lr=0.01, attack_type=attack)
                        mal_states.append({k: v.detach().clone() for k, v in st.items()})
                mal_counts = [int(np.mean(ben_counts))] * len(mal_states) if ben_counts else [counts[cid] for cid in mal_ids]
            elif mal_ids and attack == "none":
                for cid in mal_ids:
                    worker.load_state_dict(current)
                    st = train_client(worker, loaders[cid], epochs=2, lr=0.01)
                    mal_states.append({k: v.detach().clone() for k, v in st.items()})
                    mal_counts.append(counts[cid])

            states, ids, s_counts = ben_states + mal_states, ben_ids + mal_ids, ben_counts + mal_counts

            if cache is not None:
                global_model.load_state_dict(cache.step(ids, states))
            elif method == "FedAvg":   global_model = fed_avg(global_model, states, s_counts)
            elif method == "Median":   global_model.load_state_dict(coordinate_wise_median(states))
            elif method == "Krum":     global_model.load_state_dict(krum_aggregation(states, num_byzantine=f))
            elif method == "Bulyan":   global_model.load_state_dict(bulyan_aggregation(states, num_byzantine=f))
            elif method == "FLTrust":
                server_model.load_state_dict(global_model.state_dict())
                srv = train_client(server_model, root_loader, epochs=1, lr=0.01)
                global_model.load_state_dict(fltrust_aggregation(states, srv, global_model.state_dict()))
            elif method == "SWB":      global_model.load_state_dict(swb(states))
                
        return cfg, evaluate(global_model, test_loader), None
    except Exception as e:
        return cfg, -1.0, str(e)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow(["method", "attack", "corruption", "participation", "seed", "final_accuracy"])
            
    done = set()
    with open(RESULTS_FILE) as fh:
        for row in csv.DictReader(fh):
            done.add((row["method"], row["attack"], float(row["corruption"]), float(row["participation"]), int(row["seed"])))
            
    todo = []
    for m in METHODS:
        for a in ATTACKS:
            for c in CORRUPTIONS:
                if a == "none" and c > 0: continue
                if a != "none" and c == 0: continue
                for p in PARTICIPATIONS:
                    for s in SEEDS:
                        if (m, a, c, p, s) not in done:
                            todo.append((m, a, c, p, s))
                            
    print(f"{len(todo)} configs to run ({NUM_WORKERS} workers).", flush=True)
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            cfg, acc, err = fut.result()
            if err: print(f"  -> ERROR {cfg}: {err}", flush=True)
            with open(RESULTS_FILE, "a", newline="") as fh:
                csv.writer(fh).writerow([cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], f"{acc:.2f}"])
            el = (time.time() - t0) / 60; rate = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / rate if rate > 0 else -1
            print(f"[{i+1:3d}/{len(todo)}] {cfg[0]:<9} | {cfg[1]:<10} | cor={cfg[2]:<3} | part={cfg[3]:<3} | s={cfg[4]} | {acc:6.2f}% | ETA {eta:.0f}m", flush=True)
            
    print("GRID COMPLETE", flush=True)