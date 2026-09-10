# extended_all_methods.py
import csv, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import SimpleCNN, get_cifar10_loaders, train_client, evaluate, device
from aggregators import (fed_avg, coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache

ROUNDS, CHECKPOINTS = 25, {10, 25}
N_CLIENTS, N_WORKERS = 20, 6
RESULTS = "extended_all_methods_results.csv"
METHODS = ["FedAvg", "Median", "Krum", "Bulyan", "FLTrust", "SWB"]

class FlexibleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        side = img_size // 4
        self.fc1 = nn.Linear(64 * side * side, 128)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def _cifar100():
    tf = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    tr = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=tf)
    targets = np.array(tr.targets)
    idx = [[] for _ in range(N_CLIENTS)]
    for c in range(100):
        ic = np.where(targets == c)[0]; np.random.shuffle(ic)
        cuts = (np.cumsum(np.random.dirichlet(np.repeat(0.5, N_CLIENTS))) * len(ic)).astype(int)[:-1]
        for i, p in enumerate(np.split(ic, cuts)): idx[i].extend(p)
    loaders = [DataLoader(Subset(tr, p), batch_size=32, shuffle=True) for p in idx]
    test = DataLoader(te, batch_size=128, shuffle=False)
    rng = np.random.RandomState(1234)
    root = DataLoader(Subset(tr, rng.choice(len(tr), 1000, replace=False).tolist()),
                      batch_size=32, shuffle=True)
    return loaders, test, root

_cache = {}
def _worker_init():
    global _cache
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(123)
    cl, tl, rl = get_cifar10_loaders(num_clients=N_CLIENTS, alpha=0.5)
    _cache["cifar10"]  = (cl, tl, rl, 10,  lambda: SimpleCNN().to(device))
    cl2, tl2, rl2 = _cifar100()
    _cache["cifar100"] = (cl2, tl2, rl2, 100, lambda: FlexibleCNN(3, 100, 32).to(device))

def _run_one(cfg):
    ds, method, attack, cor, part, seed = cfg
    try:
        loaders, test_loader, root_loader, n_cls, factory = _cache[ds]
        counts = [len(l.dataset) for l in loaders]
        torch.manual_seed(seed); np.random.seed(seed)
        n_sample = max(int(N_CLIENTS * part), 2) + (1 if method == "Bulyan" else 0)
        byz = set(range(int(N_CLIENTS * cor))) if cor > 0 else set()
        f = int(n_sample * cor)
        trim = min(0.45, cor + 0.05)
        model, worker = factory(), factory()
        server = factory() if method == "FLTrust" else None
        cache = DelayedMomentumCache(model, N_CLIENTS,
                lambda s: swb_aggregation(s, trim_ratio=trim)) if method == "SWB-DM" else None
        accs = {}
        for r in range(1, ROUNDS + 1):
            sampled = np.random.choice(N_CLIENTS, n_sample, replace=False)
            current = model.state_dict()
            states, ids, s_counts = [], [], []
            for cid in sampled:
                worker.load_state_dict(current)
                att = attack if (cor > 0 and cid in byz) else None
                st = train_client(worker, loaders[cid], epochs=2, lr=0.01,
                                  attack_type=att, num_classes=n_cls)
                states.append({k: v.detach().clone() for k, v in st.items()})
                ids.append(cid); s_counts.append(counts[cid])
            if cache is not None:      model.load_state_dict(cache.step(ids, states))
            elif method == "FedAvg":   model = fed_avg(model, states, s_counts)
            elif method == "Median":   model.load_state_dict(coordinate_wise_median(states))
            elif method == "Krum":     model.load_state_dict(krum_aggregation(states, num_byzantine=f))
            elif method == "Bulyan":   model.load_state_dict(bulyan_aggregation(states, num_byzantine=f))
            elif method == "FLTrust":
                server.load_state_dict(current)
                srv = train_client(server, root_loader, epochs=1, lr=0.01)
                model.load_state_dict(fltrust_aggregation(states, srv, current))
            elif method == "SWB":      model.load_state_dict(swb_aggregation(states, trim_ratio=trim))
            if r in CHECKPOINTS:
                accs[r] = evaluate(model, test_loader)
        return cfg, accs, None
    except Exception as e:
        return cfg, {}, str(e)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(RESULTS):
        with open(RESULTS, "w", newline="") as fh:
            csv.writer(fh).writerow(["dataset","method","attack","corruption",
                                     "participation","seed","acc_at_round10","acc_at_round25"])
    done = set()
    with open(RESULTS) as fh:
        for row in csv.DictReader(fh):
            done.add((row["dataset"], row["method"], row["attack"],
                      float(row["corruption"]), float(row["participation"]), int(row["seed"])))
    pairs = [("none",0.0)] + [(a,c) for a in ["label_flip","sign_flip","gaussian","alie","ipm"]
                              for c in [0.1,0.2,0.3]]
    todo = []
    for m in METHODS:
        for attack, cor in pairs:
            for part in [0.5, 0.1]:
                for s in [42, 7]:
                    if ("cifar10", m, attack, cor, part, s) not in done:
                        todo.append(("cifar10", m, attack, cor, part, s))
        for attack, cor in [("none",0.0),("label_flip",0.2),("sign_flip",0.2),("gaussian",0.2)]:
            for s in [42, 7]:
                if ("cifar100", m, attack, cor, 0.5, s) not in done:
                    todo.append(("cifar100", m, attack, cor, 0.5, s))
    print(f"{len(todo)} configs to run ({N_WORKERS} workers).", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            cfg, accs, err = fut.result()
            if err: print(f"ERROR {cfg}: {err}", flush=True)
            if accs:
                with open(RESULTS, "a", newline="") as fh:
                    csv.writer(fh).writerow([cfg[0],cfg[1],cfg[2],cfg[3],cfg[4],cfg[5],
                                             f"{accs.get(10,-1):.2f}", f"{accs.get(25,-1):.2f}"])
            el = (time.time()-t0)/60; rate = (i+1)/el if el > 0 else 0
            print(f"[{i+1:3d}/{len(todo)}] {cfg[0]} | {cfg[1]:<8} | {cfg[2]:<10} | "
                  f"p={cfg[4]} | s={cfg[5]} | r10={accs.get(10,-1):.2f} r25={accs.get(25,-1):.2f} | "
                  f"ETA {(len(todo)-i-1)/rate if rate else -1:.0f}m", flush=True)
    print("EXTENSION COMPLETE", flush=True)