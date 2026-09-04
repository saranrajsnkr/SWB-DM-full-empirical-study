# tiered-depth generalization checks on CIFAR-100 and FEMNIST
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
from common import train_client, evaluate, device
from aggregators import (fed_avg, coordinate_wise_median, krum_aggregation,
                         bulyan_aggregation, swb_aggregation)
from fltrust_corrected import fltrust_aggregation
from cache import DelayedMomentumCache

NUM_CLIENTS, NUM_ROUNDS, NUM_WORKERS = 20, 10, 6
RESULTS_FILE = "generalization_results.csv"
_cache = {}

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

def _build(name):
    if name == "cifar100":
        cls, stats, spec = (torchvision.datasets.CIFAR100,
                            ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                            dict(in_c=3, n_cls=100, img=32))
    else:
        cls, stats, spec = (torchvision.datasets.EMNIST,
                            ((0.1307,), (0.3081,)),
                            dict(in_c=1, n_cls=62, img=28))
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    kw = {"split": "byclass"} if name == "femnist" else {}
    trainset = cls(root="./data", train=True, download=True, transform=tf, **kw)
    testset = cls(root="./data", train=False, download=True, transform=tf, **kw)
    targets = np.array(trainset.targets)
    idx = [[] for _ in range(NUM_CLIENTS)]
    for c in range(spec["n_cls"]):
        ic = np.where(targets == c)[0]; np.random.shuffle(ic)
        cuts = (np.cumsum(np.random.dirichlet(np.repeat(0.5, NUM_CLIENTS))) * len(ic)).astype(int)[:-1]
        for i, part in enumerate(np.split(ic, cuts)): idx[i].extend(part)
    loaders = [DataLoader(Subset(trainset, p), batch_size=32, shuffle=True) for p in idx]
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    rng = np.random.RandomState(1234)
    root_loader = DataLoader(Subset(trainset, rng.choice(len(trainset), 1000, replace=False).tolist()),
                             batch_size=32, shuffle=True)
    return loaders, test_loader, root_loader, spec

def _worker_init():
    global _cache
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(123)
    _cache["cifar100"] = _build("cifar100")
    _cache["femnist"] = _build("femnist")

def _run_one(cfg):
    ds, method, attack, cor, seed = cfg
    try:
        loaders, test_loader, root_loader, spec = _cache[ds]
        counts = [len(l.dataset) for l in loaders]
        torch.manual_seed(seed); np.random.seed(seed)
        byz = set(range(int(NUM_CLIENTS * cor))) if cor > 0 else set()
        n_sample = 11 if method == "Bulyan" else 10
        f = max(0, int(n_sample * cor))
        trim = min(0.45, cor + 0.05)
        mk = lambda: FlexibleCNN(spec["in_c"], spec["n_cls"], spec["img"]).to(device)
        model, worker = mk(), mk()
        server = mk() if method == "FLTrust" else None
        cache = DelayedMomentumCache(model, NUM_CLIENTS,
                lambda s: swb_aggregation(s, trim_ratio=trim)) if method == "SWB-DM" else None
        for r in range(1, NUM_ROUNDS + 1):
            sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
            current = model.state_dict()
            states, ids, s_counts = [], [], []
            for cid in sampled:
                worker.load_state_dict(current)
                att = attack if (cor > 0 and cid in byz) else None
                st = train_client(worker, loaders[cid], epochs=2, lr=0.01,
                                  attack_type=att, num_classes=spec["n_cls"])
                states.append({k: v.detach().clone() for k, v in st.items()})
                ids.append(cid); s_counts.append(counts[cid])
            if cache is not None:
                model.load_state_dict(cache.step(ids, states))
            elif method == "FedAvg":  model = fed_avg(model, states, s_counts)
            elif method == "Median":  model.load_state_dict(coordinate_wise_median(states))
            elif method == "Krum":    model.load_state_dict(krum_aggregation(states, num_byzantine=f))
            elif method == "Bulyan":  model.load_state_dict(bulyan_aggregation(states, num_byzantine=f))
            elif method == "FLTrust":
                server.load_state_dict(current)
                srv = train_client(server, root_loader, epochs=1, lr=0.01)
                model.load_state_dict(fltrust_aggregation(states, srv, current))
            elif method == "SWB":     model.load_state_dict(swb_aggregation(states, trim_ratio=trim))
        return cfg, evaluate(model, test_loader), None
    except Exception as e:
        return cfg, -1.0, str(e)

if __name__ == "__main__":
        # sequential pre-download: prevents parallel workers from racing on dataset extraction
    for tr in (True, False):
        torchvision.datasets.CIFAR100(root="./data", train=tr, download=True, transform=transforms.ToTensor())
        torchvision.datasets.EMNIST(root="./data", split="byclass", train=tr, download=True, transform=transforms.ToTensor())
        
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow(["dataset", "method", "attack", "corruption", "seed", "final_accuracy"])
    done = set()
    with open(RESULTS_FILE) as fh:
        for row in csv.DictReader(fh):
            done.add((row["dataset"], row["method"], row["attack"], float(row["corruption"]), int(row["seed"])))
    todo = []
    for ds in ["cifar100", "femnist"]:
        for m in ["FedAvg", "Median", "Krum", "Bulyan", "FLTrust", "SWB", "SWB-DM"]:
            for attack, cor in [("none", 0.0), ("label_flip", 0.2), ("sign_flip", 0.2), ("gaussian", 0.2)]:
                for s in [42, 7]:
                    if (ds, m, attack, cor, s) not in done:
                        todo.append((ds, m, attack, cor, s))
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
            print(f"[{i+1:3d}/{len(todo)}] {cfg[0]} | {cfg[1]:<8} | {cfg[2]:<10} | s={cfg[4]} | "
                  f"{acc:.2f}% | ETA {(len(todo)-i-1)/rate if rate else -1:.0f}m", flush=True)
    print("GENERALIZATION COMPLETE", flush=True)