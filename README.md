# Hypergame-Aware AI Defense for IoT Perception Attacks
### IEEE Transactions on Information Forensics and Security — Submission

---

## Overview

This repository contains the full simulation code, raw data, and figure generation scripts accompanying the paper *"Hypergame-Aware AI Defense for IoT Perception Attacks"*. The framework integrates $k$-level Hypergame Nash Equilibrium, a deception-aware particle filter, and adversarial meta-learning to defend a 350-device smart hospital IoT network against perception attacks.

---

## Repository Structure

```
├── Healthcare.py          # Full simulation (Eq. 1–39)
├── raw_metrics.
├── simulation_raw.npz         # Main results  — 2500 runs (50 seeds × 50 ep)
└── README.md                      # This file
```

---

## Requirements

```bash
pip install numpy scipy matplotlib
```

Python 3.9+ recommended. No GPU required.

---

## Reproducing the Results

**Run the full simulation (50 seeds):**
```bash
python Heatlhcare.py
```
Output: `simulation_raw.npz`, `simulation_results.json`

Output: PNG figures in `/outputs/`

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Devices / Patients | 350 / 50 | Smart hospital topology |
| Seeds × Episodes × Steps | 50 × 50 × 200 | 500,000 total time steps |
| Perturbation budget $\varepsilon$ | 0.15 | $\ell_2$-norm bound |
| Base attack probability $P_{\text{base}}$ | 0.30 | Main configuration |
| Reasoning depth $k$ | 2 | Both players |
| Particles per device | 20–100 | Adaptive SIR filter |

Full parameter tables are in [`supplementary_parameters.pdf`](Supplementary_parameters.pdf).

## Supplementary Material

[`supplementary_parameters.pdf`](Supplementary_parameters.pdf) contains tables covering all simulation parameters, patient physiological bounds, device criticality weights, particle filter settings, and meta-learner hyperparameters. 

---

## Reproducibility

All results are deterministic given the seed. Base seed: `BASE_SEED = 42`. Seed $i$ uses `numpy.random.default_rng(42 + i)`.
