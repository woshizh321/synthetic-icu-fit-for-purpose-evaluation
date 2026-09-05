"""PREFLIGHT-2B2 formal run script for GaussianCopula and CTGAN (SDV library).

Produces a complete run directory per docs/preflight2b2/PREFLIGHT2B2_PROTOCOL.md #10:
config.json, environment.txt, git_commit.txt, training_data_hash.txt, training.log,
resource_preflight.json, checkpoint_not_applicable, synthetic_contract.parquet,
synthetic_restored.parquet, synthetic_manifest.json, run_integrity_qc.json, RUN_STATUS.

Usage:
  python run_formal_sdv.py <MODEL: GaussianCopula|CTGAN> <DATA_PARQUET> <RUN_DIR> <SEED> <EPOCHS>
"""
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

MODEL = sys.argv[1]
DATA_PARQUET = sys.argv[2]
RUN_DIR = sys.argv[3]
SEED = int(sys.argv[4])
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 300
N_SYNTHETIC = 40745

os.makedirs(RUN_DIR, exist_ok=True)

VARS = ["heart_rate", "sbp", "dbp", "map", "respiratory_rate", "spo2", "temperature_c",
        "wbc", "hemoglobin", "platelet_count", "creatinine", "bun_urea", "sodium", "potassium",
        "glucose", "bicarbonate", "chloride", "lactate", "ph_arterial", "pao2", "paco2"]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

t_start = time.time()
run_status = "RUNNING"
error_info = None

try:
    # ---- data + hash ----
    with open(DATA_PARQUET, "rb") as fh:
        data_hash = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(RUN_DIR, "training_data_hash.txt"), "w") as fh:
        fh.write(data_hash + "\n")

    df = pd.read_parquet(DATA_PARQUET)
    idcol = "stay_id" if "stay_id" in df.columns else "CaseID"
    df = df.set_index(idcol)
    assert df.shape[1] == 150, f"expected 150 contract fields after dropping id, got {df.shape[1]}"

    BINARY_COLS = ["sex_male", "mortality_7d_post_landmark"] + [c for c in df.columns if c.endswith("_observed_any")]
    for c in BINARY_COLS:
        df[c] = df[c].astype(int)

    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    for c in BINARY_COLS:
        meta.update_column(column_name=c, sdtype="categorical")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- resource preflight ----
    resource_pre = {"seed": SEED, "model": MODEL, "device": device_str}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        resource_pre.update(total_vram_gib=round(total_b / 1024**3, 3),
                             free_vram_gib=round(free_b / 1024**3, 3))
    with open(os.path.join(RUN_DIR, "resource_preflight.json"), "w") as fh:
        json.dump(resource_pre, fh, indent=2)

    # ---- config ----
    with open(os.path.abspath(__file__), "rb") as fh:
        script_hash = hashlib.sha256(fh.read()).hexdigest()
    config = {"generator": MODEL, "seed": SEED, "epochs": EPOCHS if MODEL == "CTGAN" else "n/a (closed-form)",
              "n_synthetic": N_SYNTHETIC, "device": device_str, "training_data_sha256": data_hash,
              "library": "sdv", "library_version": __import__("sdv").__version__,
              "run_script": "run_formal_sdv.py", "run_script_sha256": script_hash,
              "run_script_version_note": "post-B2-run1 fix: explicit synth._set_random_state(SEED) added before sample() (SDV FIXED_RNG_SEED defect)"}
    with open(os.path.join(RUN_DIR, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    with open(os.path.join(RUN_DIR, "environment.txt"), "w") as fh:
        fh.write(f"python={platform.python_version()}\n")
        fh.write(f"torch={torch.__version__}\n")
        fh.write(f"cuda_available={torch.cuda.is_available()}\n")
        fh.write(f"sdv={__import__('sdv').__version__}\n")
        fh.write(f"hostname={platform.node()}\n")

    with open(os.path.join(RUN_DIR, "git_commit.txt"), "w") as fh:
        fh.write("N/A -- SDV installed via pip (not a git clone); library_version pinned "
                  f"to sdv=={__import__('sdv').__version__} in config.json is the provenance anchor.\n")

    with open(os.path.join(RUN_DIR, "checkpoint_not_applicable"), "w") as fh:
        fh.write("SDV synthesizers (GaussianCopulaSynthesizer, CTGANSynthesizer) do not expose "
                  "an intermediate checkpoint API in this pipeline; the fitted synthesizer object "
                  "itself is the only persistable state, and is not required for reproducibility "
                  "since seed + config + data hash fully determine the run.\n")

    # ---- fit + sample ----
    t0 = time.time()
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if MODEL == "GaussianCopula":
        synth = GaussianCopulaSynthesizer(meta)
    elif MODEL == "CTGAN":
        use_cuda = torch.cuda.is_available()
        synth = CTGANSynthesizer(meta, epochs=EPOCHS, batch_size=500,
                                  generator_dim=(256, 256), discriminator_dim=(256, 256),
                                  embedding_dim=128, generator_lr=2e-4, discriminator_lr=2e-4,
                                  cuda=use_cuda)
    else:
        raise ValueError(MODEL)

    synth.fit(df.reset_index(drop=True))
    fit_s = time.time() - t0

    # BUG FIX (discovered B2, run 1): SDV's BaseSingleTableSynthesizer.sample() silently
    # defaults to a FIXED internal RNG seed (FIXED_RNG_SEED=73251 in sdv/single_table/base.py)
    # unless _set_random_state() is explicitly called first. Without this call, every run
    # regardless of the SEED argument produces byte-identical synthetic output (confirmed
    # empirically: seed42 and seed43 GaussianCopula runs were SHA-256-identical before this
    # fix). This is an implementation defect in THIS script, not an intrinsic SDV limitation
    # or a scientific finding -- it is fixed here, not disclosed-and-retained, because it
    # would otherwise silently violate the seed-integrity requirement (Gate B2-G) for the
    # entire formal matrix.
    synth._set_random_state(SEED)

    t0 = time.time()
    sample = synth.sample(N_SYNTHETIC)
    sample_s = time.time() - t0

    peak_vram_gib = (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None
    peak_ram_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    with open(os.path.join(RUN_DIR, "training.log"), "w") as fh:
        fh.write(f"model={MODEL} seed={SEED} epochs={EPOCHS}\nfit_seconds={fit_s:.2f}\n"
                 f"sample_seconds={sample_s:.2f}\npeak_vram_gib={peak_vram_gib}\npeak_ram_mb={peak_ram_mb:.1f}\n")

    # ---- raw contract output (legitimate decoding only, no repair) ----
    for c in BINARY_COLS:
        sample[c] = sample[c].round().clip(0, 1).astype(int)
    sample = sample[list(df.columns)]
    sample.to_parquet(os.path.join(RUN_DIR, "synthetic_contract.parquet"))

    # ---- DEC-008 restoration (deterministic decoding, not repair) ----
    restored = sample.copy()
    for v in VARS:
        mask = restored[f"{v}_observed_any"] == 0
        for suffix in ["first", "last", "min", "max", "median"]:
            restored.loc[mask, f"{v}_{suffix}"] = np.nan
    restored.to_parquet(os.path.join(RUN_DIR, "synthetic_restored.parquet"))

    # ---- integrity QC ----
    qc = {}
    qc["schema"] = {
        "n_rows": int(len(sample)), "n_rows_expected": N_SYNTHETIC, "n_rows_match": len(sample) == N_SYNTHETIC,
        "n_cols": int(sample.shape[1]), "n_cols_expected": 150, "n_cols_match": sample.shape[1] == 150,
        "columns_match_order": list(sample.columns) == list(df.columns),
    }
    nan_total = int(sample.isna().sum().sum())
    num_cols = sample.select_dtypes(include=[np.number]).columns
    inf_total = int(np.isinf(sample[num_cols].values).sum())
    invalid_binary = 0
    for c in BINARY_COLS:
        invalid_binary += int((~sample[c].isin([0, 1])).sum())
    qc["numerical_integrity"] = {"nan_total_raw_contract": nan_total, "inf_total": inf_total,
                                  "invalid_binary_total": invalid_binary}

    outcome_col = "mortality_7d_post_landmark"
    n_pos = int(sample[outcome_col].sum())
    n_neg = int(len(sample) - n_pos)
    qc["outcome"] = {"prevalence_pct": round(100 * n_pos / len(sample), 4), "n_events": n_pos,
                      "n_survivors": n_neg, "single_class_outcome": (n_pos == 0 or n_neg == 0)}

    dens_cols = [c for c in df.columns if c.endswith("_observed_hour_density")]
    dens_oob = int(((sample[dens_cols] < -1e-6) | (sample[dens_cols] > 1 + 1e-6)).sum().sum())
    mask_density_incon = 0
    for v in VARS:
        oa = sample[f"{v}_observed_any"]
        dens = sample[f"{v}_observed_hour_density"]
        mask_density_incon += int(((oa == 0) & (dens > 1e-6)).sum())
        mask_density_incon += int(((oa == 1) & (dens <= 1e-6)).sum())
    qc["observation_process"] = {"density_out_of_range_count": dens_oob,
                                  "density_out_of_range_cells_total": len(sample) * len(dens_cols),
                                  "mask_density_inconsistency_count": mask_density_incon}

    order_all = 0
    order_observed_only = 0
    for v in VARS:
        f, l, mn, mx, med = (sample[f"{v}_first"], sample[f"{v}_last"], sample[f"{v}_min"],
                              sample[f"{v}_max"], sample[f"{v}_median"])
        viol_mask = (mn > med) | (med > mx) | (mn > mx) | (f < mn) | (f > mx) | (l < mn) | (l > mx)
        order_all += int(viol_mask.sum())
        oa = sample[f"{v}_observed_any"] == 1
        order_observed_only += int((viol_mask & oa).sum())
    qc["summary_ordering"] = {"violations_all_rows": order_all, "violations_observed_any_rows": order_observed_only}

    with open(os.path.join(RUN_DIR, "run_integrity_qc.json"), "w") as fh:
        json.dump(qc, fh, indent=2)

    # ---- manifest ----
    contract_hash = hashlib.sha256(open(os.path.join(RUN_DIR, "synthetic_contract.parquet"), "rb").read()).hexdigest()
    restored_hash = hashlib.sha256(open(os.path.join(RUN_DIR, "synthetic_restored.parquet"), "rb").read()).hexdigest()

    run_integrity_ok = (qc["schema"]["n_rows_match"] and qc["schema"]["n_cols_match"]
                         and qc["schema"]["columns_match_order"] and qc["numerical_integrity"]["inf_total"] == 0
                         and qc["numerical_integrity"]["invalid_binary_total"] == 0)
    run_status = "OUTCOME_COLLAPSE" if qc["outcome"]["single_class_outcome"] else ("COMPLETE" if run_integrity_ok else "QC_FAILED")

    manifest = {
        "run_id": os.path.basename(RUN_DIR.rstrip("/")), "generator": MODEL,
        "generator_family": "statistical" if MODEL == "GaussianCopula" else "GAN",
        "generator_version": f"sdv=={__import__('sdv').__version__}", "git_commit": "N/A (pip package, see git_commit.txt)",
        "representation": "A", "seed": SEED, "real_train_n": int(len(df)), "synthetic_n": int(len(sample)),
        "training_data_hash": data_hash, "generator_contract_hash": data_hash, "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "environment_hash": hashlib.sha256(open(os.path.join(RUN_DIR, "environment.txt"), "rb").read()).hexdigest(),
        "start_time": t_start, "end_time": time.time(), "wall_time_sec": round(time.time() - t_start, 2),
        "device": device_str, "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a (CPU-only run)",
        "total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else "n/a",
        "peak_vram_gib": round(peak_vram_gib, 3) if peak_vram_gib is not None else "n/a (CPU-only)",
        "peak_ram_gib": round(peak_ram_mb / 1024, 3), "checkpoint_hash": "N/A (checkpoint_not_applicable)",
        "synthetic_contract_hash": contract_hash, "synthetic_restored_hash": restored_hash,
        "synthetic_outcome_prevalence": qc["outcome"]["prevalence_pct"], "synthetic_event_n": qc["outcome"]["n_events"],
        "schema_valid": bool(run_integrity_ok), "single_class_outcome": bool(qc["outcome"]["single_class_outcome"]),
        "density_violation_rate": round(dens_oob / (len(sample) * len(dens_cols)), 6),
        "ordering_violation_rate": round(order_all / (len(sample) * len(VARS)), 6),
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
