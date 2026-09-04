# SWB-DM extended-rounds rerun
#
# The main grid runs everyone for 10 rounds. SWB-DM's cache mechanism has
# already been shown (standalone checks) to need 20-30 rounds to warm up,
# especially at part=0.1 where only 2 of 20 clients refresh their cache
# entry per round. This script reruns SWB-DM ONLY, same attack x corruption
# x participation x seed grid as grid_main.py, but at NUM_ROUNDS=25, so we
# can report a warm-up-adjusted comparison alongside the 10-round numbers
# rather than letting the 10-round SWB-DM numbers stand unqualified.
import csv, os, time
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import swb_aggregation
from cache import DelayedMomentumCache
from attacks import generate_adaptive_attacks

NUM_CLIENTS = 20
NUM_ROUNDS = 25          # up from 10 in the main grid
METHOD = "SWB-DM"
ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "alie", "ipm"]
CORRUPTIONS = [0.0, 0.1, 0.2, 0.3]
PARTICIPATIONS = [0.5, 0.1]
SEEDS = [42, 7]
RESULTS_FILE = "swb_dm_extended_results.csv"
NUM_WORKERS = 4
_cache = {}


def _worker_init():
    global _cache
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(123)  # same fixed partition seed as grid_main.py
    cl, tl, rl = get_cifar10_loaders(num_clients=NUM_CLIENTS, alpha=0.5)
    _cache.update(loaders=cl, test=tl, root=rl, counts=[len(l.dataset) for l in cl])


def assumed_f(n, corr):
    return max(0, int(n * corr))


def trim_for(corr):
    return min(0.45, corr + 0.05)


def _run_one(cfg):
    attack, corr, part, seed = cfg
    try:
        loaders, test_loader = _cache["loaders"], _cache["test"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        byz = set(range(int(NUM_CLIENTS * corr))) if corr > 0 else set()
        n_sample = max(1, int(NUM_CLIENTS * part))
        trim = trim_for(corr)
        swb = lambda s: swb_aggregation(s, trim_ratio=trim)

        global_model = SimpleCNN().to(device)
        worker = SimpleCNN().to(device)
        cache = DelayedMomentumCache(global_model, NUM_CLIENTS, swb)

        # Checkpoint at round 10 too, so we can directly compare against
        # the main grid's 10-round number from the SAME run, not just the
        # final extended-round number.
        acc_at_10 = None

        for r in range(1, NUM_ROUNDS + 1):
            sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
            current = global_model.state_dict()
            mal_ids = [c for c in sampled if c in byz]
            ben_ids = [c for c in sampled if c not in byz]

            ben_states = []
            for cid in ben_ids:
                worker.load_state_dict(current)
                st = train_client(worker, loaders[cid], epochs=2, lr=0.01)
                ben_states.append({k: v.detach().clone() for k, v in st.items()})

            mal_states = []
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
            elif mal_ids and attack == "none":
                for cid in mal_ids:
                    worker.load_state_dict(current)
                    st = train_client(worker, loaders[cid], epochs=2, lr=0.01)
                    mal_states.append({k: v.detach().clone() for k, v in st.items()})

            states = ben_states + mal_states
            ids = ben_ids + mal_ids
            global_model.load_state_dict(cache.step(ids, states))

            if r == 10:
                acc_at_10 = evaluate(global_model, test_loader)

        acc_final = evaluate(global_model, test_loader)
        return cfg, acc_at_10, acc_final, None
    except Exception as e:
        return cfg, None, -1.0, str(e)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow(["method", "attack", "corruption", "participation", "seed",
                                     "acc_at_round10", "acc_at_round25"])

    done = set()
    with open(RESULTS_FILE) as fh:
        for row in csv.DictReader(fh):
            done.add((row["attack"], float(row["corruption"]), float(row["participation"]), int(row["seed"])))

    todo = []
    for a in ATTACKS:
        for c in CORRUPTIONS:
            if a == "none" and c > 0:
                continue
            if a != "none" and c == 0:
                continue
            for p in PARTICIPATIONS:
                for s in SEEDS:
                    if (a, c, p, s) not in done:
                        todo.append((a, c, p, s))

    print(f"{len(todo)} SWB-DM configs to run at {NUM_ROUNDS} rounds ({NUM_WORKERS} workers).", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            cfg, acc10, acc_final, err = fut.result()
            if err:
                print(f"  -> ERROR {cfg}: {err}", flush=True)
            with open(RESULTS_FILE, "a", newline="") as fh:
                csv.writer(fh).writerow([METHOD, cfg[0], cfg[1], cfg[2], cfg[3],
                                         f"{acc10:.2f}" if acc10 is not None else "NA",
                                         f"{acc_final:.2f}"])
            el = (time.time() - t0) / 60
            rate = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / rate if rate > 0 else -1
            print(f"[{i+1:3d}/{len(todo)}] {cfg[0]:<10} | cor={cfg[1]:<3} | part={cfg[2]:<3} | s={cfg[3]} | "
                  f"r10={acc10:.2f}% -> r25={acc_final:.2f}% | ETA {eta:.0f}m", flush=True)

    print("SWB-DM EXTENDED RERUN COMPLETE", flush=True)
