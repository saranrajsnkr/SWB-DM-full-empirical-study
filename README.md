# SWB-DM: Sliced-Wasserstein Barycenter Aggregation with Delayed Momentum
**Byzantine-Robust Federated Learning under Partial Client Participation**

## Overview
Standard Byzantine-robust aggregators (Krum, Bulyan, Median) assume the sampled client subset is large enough to keep the in-sample Byzantine fraction below their breakdown point. Under realistic **partial client participation** (e.g., sampling 10% of clients per round), random sampling frequently creates majority-Byzantine subsets, causing catastrophic constant-prediction collapse. 

Building directly on the **Delayed Momentum Aggregation (DeMoA)** principle (Yamada et al.), this repository implements **SWB-DM**. It combines a full-pool delayed-momentum cache (which restores the population-level Byzantine fraction) with **SWB**, a Sliced-Wasserstein/Robust-Mean-Estimation aggregator that applies trimmed 1-D Wasserstein barycenters under random orthogonal projections.

## Key Empirical Findings
This repository contains a fully reproducible, 448+ configuration empirical study validating the following:
1. **Collapse Prevention:** At 30% corruption and 50% participation, non-cached baselines collapse to ~10% accuracy under IPM/ALIE attacks. SWB-DM survives by anchoring to the full-pool cache.
2. **FLTrust's Mechanistic Immunity:** Multi-seed trust-weight traces prove FLTrust's ReLU-clipped cosine filter assigns exactly zero weight to sign-flip attackers.
3. **The Warm-Up Trade-off:** SWB-DM pays a quantifiable cold-start penalty at low participation (10%), requiring ~25 rounds to recover, but ultimately prevents the death-spirals that destroy sample-only aggregators.
4. **Krum's Scale Inefficiency:** At 500 clients, Krum collapses to 10% *even with zero attackers*, because single-client selection discards 98% of updates when clients hold sparse data.
5. **Theory Validation:** Empirical one-round aggregation error under ALIE and IPM strictly tracks the theoretical $E_0 + C\sqrt{\beta}$ robust-estimation bound.

## Repository Structure
*   `common.py`: Shared CNN architectures, Dirichlet non-IID partitioning, and client training loops.
*   `aggregators.py`: Implementations of FedAvg, Median, Krum, Bulyan, and SWB.
*   `fltrust_corrected.py`: FLTrust with proper norm-clipping.
*   `cache.py`: The Delayed Momentum full-pool cache mechanism.
*   `attacks.py`: Naive attacks (label/sign/gaussian) and adaptive attacks (ALIE, IPM).
*   `grid_main.py`: The 448-config main experimental grid.
*   `generalization.py`: CIFAR-100 and FEMNIST tiered-depth generalization checks.
*   `scalability.py`: 500-client wall-clock and communication cost analysis.

## How to Run
Requires PyTorch, TorchVision, and NumPy.
```bash
python grid_main.py
python scalability.py