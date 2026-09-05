"""PREFLIGHT-2B2 formal run script for TabDDPM (corrected implementation, full budget).

Carries forward the B1-07 density fix (QuantileTransformer) and B1-06 training schedule
(AdamW + linear LR annealing + EMA) UNCHANGED. Ordering violations are intrinsic and are
NOT repaired (DEC-011) -- quantified only. Full DEC-013 budget: epochs=100, num_timesteps=1000.

Usage:
  python run_formal_tabddpm.py <TABDDPM_REPO> <DATA_PARQUET> <RUN_DIR> <SEED> <N_EPOCHS> <NUM_TIMESTEPS>
"""
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer

TABDDPM_REPO = sys.argv[1]
DATA_PARQUET = sys.argv[2]
RUN_DIR = sys.argv[3]
SEED = int(sys.argv[4])
N_EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 100
NUM_TIMESTEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
N_SYNTHETIC = 40745

os.makedirs(RUN_DIR, exist_ok=True)

sys.path.insert(0, TABDDPM_REPO)
from tab_ddpm.modules import MLPDiffusion
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARS = ["heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temperature_c",
        "wbc", "hemoglobin", "platelet_count", "creatinine", "bun_urea", "sodium", "potassium",
        "glucose", "bicarbonate", "chloride", "lactate", "ph_arterial", "pao2", "paco2"]

t_start = time.time()
run_status = "RUNNING"

try:
    with open(DATA_PARQUET, "rb") as fh:
        data_hash = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(RUN_DIR, "training_data_hash.txt"), "w") as fh:
        fh.write(data_hash + "\n")

    git_commit = subprocess.run(["git", "-C", TABDDPM_REPO, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    with open(os.path.join(RUN_DIR, "git_commit.txt"), "w") as fh:
        fh.write(f"tab-ddpm repo commit: {git_commit}\n")

    df = pd.read_parquet(DATA_PARQUET)
    idcol = "stay_id" if "stay_id" in df.columns else "CaseID"
    df = df.set_index(idcol)
    assert df.shape[1] == 150, f"expected 150 contract fields after dropping id, got {df.shape[1]}"

    CAT_COLS = ["sex_male", "mortality_7d_post_landmark"] + [c for c in df.columns if c.endswith("_observed_any")]
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS]

    resource_pre = {"seed": SEED, "model": "TabDDPM_corrected", "device": str(device)}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        resource_pre.update(total_vram_gib=round(total_b / 1024**3, 3), free_vram_gib=round(free_b / 1024**3, 3))
    with open(os.path.join(RUN_DIR, "resource_preflight.json"), "w") as fh:
        json.dump(resource_pre, fh, indent=2)

    config = {"generator": "TabDDPM_corrected", "seed": SEED, "n_epochs": N_EPOCHS, "num_timesteps": NUM_TIMESTEPS,
              "d_layers": [256, 256], "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
              "ema_decay": 0.999, "scheduler": "cosine",
              "numeric_normalization": "QuantileTransformer(output_distribution=normal)",
              "n_synthetic": N_SYNTHETIC, "device": str(device), "training_data_sha256": data_hash,
              "tab_ddpm_git_commit": git_commit}
    with open(os.path.join(RUN_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    with open(os.path.join(RUN_DIR, "environment.txt"), "w") as fh:
        fh.write(f"python={platform.python_version()}\ntorch={torch.__version__}\n"
                 f"cuda_available={torch.cuda.is_available()}\nhostname={platform.node()}\n")

    with open(os.path.join(RUN_DIR, "checkpoint_not_applicable"), "w") as fh:
        fh.write("No intermediate checkpoint saved in this run script; final EMA-weighted model "
                  "state is not persisted separately since seed + config + data hash fully determine "
                  "the run and reproducibility does not depend on mid-training checkpoints here.\n")

    t0 = time.time()
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    X_num = df[NUM_COLS].values.astype(np.float64)
    qt = QuantileTransformer(output_distribution="normal",
                              n_quantiles=max(min(len(df) // 30, 1000), 10),
                              subsample=int(1e9), random_state=SEED)
    X_num_n = qt.fit_transform(X_num)

    X_cat = df[CAT_COLS].values.astype(int)
    num_classes = np.array([2] * len(CAT_COLS))
    X_glue = np.concatenate([X_num_n, X_cat.astype(np.float64)], axis=1).astype(np.float32)
    d_in = len(NUM_COLS) + int(num_classes.sum())

    denoise_fn = MLPDiffusion(d_in=d_in, num_classes=0, is_y_cond=False,
                               rtdl_params={"d_layers": [256, 256], "dropout": 0.0}).to(device)
    diffusion = GaussianMultinomialDiffusion(
        num_classes=num_classes, num_numerical_features=len(NUM_COLS),
        denoise_fn=denoise_fn, num_timesteps=NUM_TIMESTEPS, scheduler="cosine", device=device,
    ).to(device)
    diffusion.train()

    LR = 1e-3
    opt = torch.optim.AdamW(diffusion.parameters(), lr=LR, weight_decay=1e-4)
    ema_model = copy.deepcopy(diffusion._denoise_fn)
    for p in ema_model.parameters():
        p.detach_()

    def update_ema(target, source, rate=0.999):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(rate).add_(sp.data, alpha=1 - rate)

    Xt = torch.tensor(X_glue, device=device)
    BATCH = 256
    n = Xt.shape[0]
    steps_per_epoch = max(1, n // BATCH)
    total_steps = N_EPOCHS * steps_per_epoch
    step = 0
    losses = []
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n, device=device)  # global seed set once above (SEED); matches B1 implementation
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            batch = Xt[idx]
            frac_done = step / max(1, total_steps)
            for pg in opt.param_groups:
                pg["lr"] = LR * (1 - frac_done)
            opt.zero_grad()
            loss_multi, loss_gauss = diffusion.mixed_loss(batch, {})
            loss = loss_multi + loss_gauss
            loss.backward()
            opt.step()
            update_ema(ema_model, diffusion._denoise_fn)
            ep_loss += float(loss.item()) * len(idx)
            step += 1
        losses.append(round(ep_loss / n, 4))

    fit_s = time.time() - t0
    if any(np.isnan(losses)) or any(np.isinf(losses)):
        raise RuntimeError(f"NaN/Inf loss encountered (scientific/model failure, not retried): {losses}")

    with open(os.path.join(RUN_DIR, "training.log"), "w") as fh:
        fh.write(f"seed={SEED} epochs={N_EPOCHS} timesteps={NUM_TIMESTEPS}\nfit_seconds={fit_s:.2f}\n"
                 f"epoch_losses={losses}\n")

    t0 = time.time()
    with torch.no_grad():
        dummy_y_dist = torch.tensor([1.0], device=device)
        x_gen, _ = diffusion.sample_all(N_SYNTHETIC, batch_size=256, y_dist=dummy_y_dist)
    sample_s = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if device.type == "cuda" else tracemalloc.get_traced_memory()[1] / 1e6
    peak_vram_gib = (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == "cuda" else None
    tracemalloc.stop()

    with open(os.path.join(RUN_DIR, "training.log"), "a") as fh:
        fh.write(f"sample_seconds={sample_s:.2f}\npeak_memory_mb={peak_mem_mb:.1f}\n")

    x_gen_np = x_gen.cpu().numpy()
    z_num = qt.inverse_transform(x_gen_np[:, :len(NUM_COLS)])
    z_cat = x_gen_np[:, len(NUM_COLS):]

    sample_df = pd.DataFrame(z_num, columns=NUM_COLS)
    for j, c in enumerate(CAT_COLS):
        sample_df[c] = z_cat[:, j].round().astype(int).clip(0, 1)
    sample_df = sample_df[list(df.columns)]
    sample_df.to_parquet(os.path.join(RUN_DIR, "synthetic_contract.parquet"))

    restored = sample_df.copy()
    for v in VARS:
        mask = restored[f"{v}_observed_any"] == 0
        for suffix in ["first", "last", "min", "max", "median"]:
            restored.loc[mask, f"{v}_{suffix}"] = np.nan
    restored.to_parquet(os.path.join(RUN_DIR, "synthetic_restored.parquet"))

    # ---- integrity QC ----
    qc = {}
    qc["schema"] = {"n_rows": int(len(sample_df)), "n_rows_expected": N_SYNTHETIC,
                     "n_rows_match": len(sample_df) == N_SYNTHETIC, "n_cols": int(sample_df.shape[1]),
                     "n_cols_expected": 150, "n_cols_match": sample_df.shape[1] == 150,
                     "columns_match_order": list(sample_df.columns) == list(df.columns)}
    nan_total = int(sample_df.isna().sum().sum())
    num_cols_all = sample_df.select_dtypes(include=[np.number]).columns
    inf_total = int(np.isinf(sample_df[num_cols_all].values).sum())
    invalid_binary = 0
    for c in CAT_COLS:
        invalid_binary += int((~sample_df[c].isin([0, 1])).sum())
    qc["numerical_integrity"] = {"nan_total_raw_contract": nan_total, "inf_total": inf_total,
                                  "invalid_binary_total": invalid_binary}

    outcome_col = "mortality_7d_post_landmark"
    n_pos = int(sample_df[outcome_col].sum())
    n_neg = int(len(sample_df) - n_pos)
    qc["outcome"] = {"prevalence_pct": round(100 * n_pos / len(sample_df), 4), "n_events": n_pos,
                      "n_survivors": n_neg, "single_class_outcome": (n_pos == 0 or n_neg == 0)}

    dens_cols = [c for c in df.columns if c.endswith("_observed_hour_density")]
    dens_oob = int(((sample_df[dens_cols] < -1e-6) | (sample_df[dens_cols] > 1 + 1e-6)).sum().sum())
    mask_density_incon = 0
    for v in VARS:
        oa = sample_df[f"{v}_observed_any"]
        dens = sample_df[f"{v}_observed_hour_density"]
        mask_density_incon += int(((oa == 0) & (dens > 1e-6)).sum())
        mask_density_incon += int(((oa == 1) & (dens <= 1e-6)).sum())
    qc["observation_process"] = {"density_out_of_range_count": dens_oob,
                                  "density_out_of_range_cells_total": len(sample_df) * len(dens_cols),
                                  "mask_density_inconsistency_count": mask_density_incon}

    order_all = 0
    order_observed_only = 0
    for v in VARS:
        f, l, mn, mx, med = (sample_df[f"{v}_first"], sample_df[f"{v}_last"], sample_df[f"{v}_min"],
                              sample_df[f"{v}_max"], sample_df[f"{v}_median"])
        viol_mask = (mn > med) | (med > mx) | (mn > mx) | (f < mn) | (f > mx) | (l < mn) | (l > mx)
        order_all += int(viol_mask.sum())
        oa = sample_df[f"{v}_observed_any"] == 1
        order_observed_only += int((viol_mask & oa).sum())
    qc["summary_ordering"] = {"violations_all_rows": order_all, "violations_observed_any_rows": order_observed_only,
                               "note": "INTRINSIC per DEC-011 -- quantified, not repaired."}

    with open(os.path.join(RUN_DIR, "run_integrity_qc.json"), "w") as fh:
        json.dump(qc, fh, indent=2)

    contract_hash = hashlib.sha256(open(os.path.join(RUN_DIR, "synthetic_contract.parquet"), "rb").read()).hexdigest()
    restored_hash = hashlib.sha256(open(os.path.join(RUN_DIR, "synthetic_restored.parquet"), "rb").read()).hexdigest()

    run_integrity_ok = (qc["schema"]["n_rows_match"] and qc["schema"]["n_cols_match"]
                         and qc["schema"]["columns_match_order"] and qc["numerical_integrity"]["inf_total"] == 0
                         and qc["numerical_integrity"]["invalid_binary_total"] == 0
                         and qc["observation_process"]["density_out_of_range_count"] == 0)
    run_status = "OUTCOME_COLLAPSE" if qc["outcome"]["single_class_outcome"] else ("COMPLETE" if run_integrity_ok else "QC_FAILED")

    manifest = {
        "run_id": os.path.basename(RUN_DIR.rstrip("/")), "generator": "TabDDPM_corrected",
        "generator_family": "diffusion", "generator_version": "yandex-research/tab-ddpm (corrected)",
        "git_commit": git_commit, "representation": "A", "seed": SEED, "real_train_n": int(len(df)),
        "synthetic_n": int(len(sample_df)), "training_data_hash": data_hash, "generator_contract_hash": data_hash,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "environment_hash": hashlib.sha256(open(os.path.join(RUN_DIR, "environment.txt"), "rb").read()).hexdigest(),
        "start_time": t_start, "end_time": time.time(), "wall_time_sec": round(time.time() - t_start, 2),
        "device": str(device), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a",
        "total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else "n/a",
        "peak_vram_gib": round(peak_vram_gib, 3) if peak_vram_gib is not None else "n/a",
        "peak_ram_gib": round(peak_mem_mb / 1024, 3), "checkpoint_hash": "N/A (checkpoint_not_applicable)",
        "synthetic_contract_hash": contract_hash, "synthetic_restored_hash": restored_hash,
        "synthetic_outcome_prevalence": qc["outcome"]["prevalence_pct"], "synthetic_event_n": qc["outcome"]["n_events"],
        "schema_valid": bool(run_integrity_ok), "single_class_outcome": bool(qc["outcome"]["single_class_outcome"]),
        "density_violation_rate": round(dens_oob / (len(sample_df) * len(dens_cols)), 6),
        "ordering_violation_rate": round(order_all / (len(sample_df) * len(VARS)), 6),
        "retry_count": 0, "run_status": run_status,
    }
    with open(os.path.join(RUN_DIR, "synthetic_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

except Exception as e:
    import traceback
    error_info = {"error": str(e)[:2000], "traceback": traceback.format_exc()[-4000:]}
    with open(os.path.join(RUN_DIR, "error.log"), "w") as fh:
        fh.write(json.dumps(error_info, indent=2))
    run_status = "MODEL_FAILURE"

with open(os.path.join(RUN_DIR, "RUN_STATUS"), "w") as fh:
    fh.write(run_status + "\n")

print(f"RUN_STATUS={run_status}")
print("DONE")
