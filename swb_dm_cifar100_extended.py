#SWB-DM extended-rounds rerun on CIFAR-100 only
#
# SWB-DM scored 9-14% on CIFAR-100 at 10 rounds -- dramatically lower than
# every other method (13-32%), while on FEMNIST it stayed close to the
# pack (77-81%). This is consistent with the cache warm-up effect already
# confirmed on CIFAR-10 (64/64 configs improved substantially from round
# 10 to round 25), compounded by CIFAR-100 being a harder 100-class task
# needing more gradient signal per round to make progress. This script
# reruns SWB-DM alone on CIFAR-100, same attack/corruption grid, at 25
# rounds instead of 10, checkpointing at round 10 for direct comparison.
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
from aggregators import swb_aggregation
from cache import DelayedMomentumCache

NUM_CLIENTS = 20
NUM_ROUNDS = 25          # up from 10
CHECKPOINT_ROUND = 10    # matches the original grid's measurement point
NUM_WORKERS = 5
RESULTS_FILE = "swb_dm_cifar100_extended_results.csv"
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


def _build():
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    spec = dict(in_c=3, n_cls=100, img=32)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=tf)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=tf)
    targets = np.array(trainset.targets)
    idx = [[] for _ in range(NUM_CLIENTS)]
    for c in range(spec["n_cls"]):
        ic = np.where(targets == c)[0]
        np.random.shuffle(ic)
        cuts = (np.cumsum(np.random.dirichlet(np.repeat(0.5, NUM_CLIENTS))) * len(ic)).astype(int)[:-1]
        for i, part in enumerate(np.split(ic, cuts)):
            idx[i].extend(part)
    loaders = [DataLoader(Subset(trainset, p), batch_size=32, shuffle=True) for p in idx]
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    return loaders, test_loader, spec


def _worker_init():
    global _cache
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(123)  # same fixed partition seed as generalization.py
    _cache["data"] = _build()


def trim_for(corr):
    return min(0.45, corr + 0.05)


def _run_one(cfg):
    attack, corr, seed = cfg
    try:
        loaders, test_loader, spec = _cache["data"]
        counts = [len(l.dataset) for l in loaders]
        torch.manual_seed(seed)
        np.random.seed(seed)
        byz = set(range(int(NUM_CLIENTS * corr))) if corr > 0 else set()
        n_sample = 10
        trim = trim_for(corr)

        mk = lambda: FlexibleCNN(spec["in_c"], spec["n_cls"], spec["img"]).to(device)
        model, worker = mk(), mk()
        cache = DelayedMomentumCache(model, NUM_CLIENTS, lambda s: swb_aggregation(s, trim_ratio=trim))

        acc_at_10 = None
        for r in range(1, NUM_ROUNDS + 1):
            sampled = np.random.choice(NUM_CLIENTS, n_sample, replace=False)
            current = model.state_dict()
            states, ids = [], []
            for cid in sampled:
                worker.load_state_dict(current)
                att = attack if (corr > 0 and cid in byz) else None
                st = train_client(worker, loaders[cid], epochs=2, lr=0.01,
                                  attack_type=att, num_classes=spec["n_cls"])
                states.append({k: v.detach().clone() for k, v in st.items()})
                ids.append(cid)
            model.load_state_dict(cache.step(ids, states))

            if r == CHECKPOINT_ROUND:
                acc_at_10 = evaluate(model, test_loader)

        acc_final = evaluate(model, test_loader)
        return cfg, acc_at_10, acc_final, None
    except Exception as e:
        return cfg, None, -1.0, str(e)


if __name__ == "__main__":
    torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transforms.ToTensor())
    torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transforms.ToTensor())

    mp.set_start_method("spawn", force=True)
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as fh:
            csv.writer(fh).writerow(["method", "attack", "corruption", "seed",
                                     "acc_at_round10", f"acc_at_round{NUM_ROUNDS}"])

    done = set()
    with open(RESULTS_FILE) as fh:
        for row in csv.DictReader(fh):
            done.add((row["attack"], float(row["corruption"]), int(row["seed"])))

    todo = [(attack, cor, s)
            for attack, cor in [("none", 0.0), ("label_flip", 0.2), ("sign_flip", 0.2), ("gaussian", 0.2)]
            for s in [42, 7]
            if (attack, cor, s) not in done]

    print(f"{len(todo)} SWB-DM/CIFAR-100 configs to run at {NUM_ROUNDS} rounds ({NUM_WORKERS} workers).", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(_run_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs)):
            cfg, acc10, acc_final, err = fut.result()
            if err:
                print(f"  -> ERROR {cfg}: {err}", flush=True)
            with open(RESULTS_FILE, "a", newline="") as fh:
                csv.writer(fh).writerow(["SWB-DM", cfg[0], cfg[1], cfg[2],
                                         f"{acc10:.2f}" if acc10 is not None else "NA",
                                         f"{acc_final:.2f}"])
            el = (time.time() - t0) / 60
            rate = (i + 1) / el if el > 0 else 0
            eta = (len(todo) - i - 1) / rate if rate > 0 else -1
            print(f"[{i+1}/{len(todo)}] {cfg[0]:<10} | cor={cfg[1]:<3} | s={cfg[2]} | "
                  f"r10={acc10:.2f}% -> r{NUM_ROUNDS}={acc_final:.2f}% | ETA {eta:.0f}m", flush=True)

    print("SWB-DM CIFAR-100 EXTENDED RERUN COMPLETE", flush=True)
