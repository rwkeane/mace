#!/usr/bin/env python3
"""Create an rMD17 split, run MACE repeatedly, and summarize the results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import yaml
from torch_geometric.datasets import MD17

from create_rmd17_dataset import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_CACHE_DIR,
    RMD17_MOLECULES,
    ensure_rmd17_raw_data,
    pyg_data_to_atoms,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one rMD17 split, train MACE repeatedly, and print aggregate "
            "loss, accuracy, timing, and memory statistics."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Required destination for train.xyz, valid.xyz, run outputs, and summaries",
    )
    parser.add_argument(
        "-k",
        "--num-runs",
        type=int,
        required=True,
        help="Number of independent training runs",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs per run",
    )
    parser.add_argument(
        "--molecule",
        choices=RMD17_MOLECULES,
        default="revised aspirin",
        help="rMD17 molecule to use (default: revised aspirin)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base MACE YAML configuration",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="PyG download/cache directory",
    )
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--valid-size", type=int, default=50)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed used once to construct the shared split",
    )
    parser.add_argument(
        "--first-run-seed",
        type=int,
        default=0,
        help="Seed for run zero; subsequent runs increment it by one",
    )
    parser.add_argument(
        "--loss-csv-dir",
        type=Path,
        default=None,
        help=(
            "If set, write per-run train and validation (held-out) loss curves "
            "as CSV files in this directory"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.train_size <= 0 or args.valid_size <= 0:
        raise ValueError("--train-size and --valid-size must be positive")
    if not args.base_config.is_file():
        raise FileNotFoundError(f"Base config does not exist: {args.base_config}")


def create_split(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.xyz"
    valid_path = output_dir / "valid.xyz"
    manifest_path = output_dir / "split.json"

    total = args.train_size + args.valid_size
    cache_dir = args.cache_dir.resolve()
    ensure_rmd17_raw_data(cache_dir, args.molecule)
    dataset = MD17(root=str(cache_dir), name=args.molecule)
    if len(dataset) < total:
        raise ValueError(
            f"{args.molecule} has {len(dataset)} configurations; {total} requested"
        )

    # Match create_rmd17_dataset.py: shuffle the first requested configurations.
    indices = np.arange(total)
    np.random.default_rng(args.split_seed).shuffle(indices)
    train_indices = indices[: args.train_size]
    valid_indices = indices[args.train_size :]

    print(
        f"Writing shared split: {len(train_indices)} train / "
        f"{len(valid_indices)} valid"
    )
    ase.io.write(
        train_path,
        [pyg_data_to_atoms(dataset[int(i)]) for i in train_indices],
        format="extxyz",
    )
    ase.io.write(
        valid_path,
        [pyg_data_to_atoms(dataset[int(i)]) for i in valid_indices],
        format="extxyz",
    )

    manifest = {
        "molecule": args.molecule,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "split_seed": args.split_seed,
        "train_indices": train_indices.tolist(),
        "valid_indices": valid_indices.tolist(),
        "train_file": str(train_path),
        "valid_file": str(valid_path),
        "energy_units": "eV",
        "forces_units": "eV/Angstrom",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return train_path, valid_path, manifest_path


def load_base_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def write_run_config(
    *,
    base_config: dict[str, Any],
    run_dir: Path,
    train_path: Path,
    valid_path: Path,
    molecule: str,
    run_index: int,
    seed: int,
    epochs: int,
) -> tuple[Path, str]:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use a fresh --output-dir."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    config = dict(base_config)
    molecule_slug = molecule.replace(" ", "_")
    name = f"validate_{molecule_slug}_{run_index:02d}"
    config.update(
        {
            "name": name,
            "seed": seed,
            "train_file": str(train_path),
            "valid_file": str(valid_path),
            "max_num_epochs": epochs,
            "eval_interval": 1,
            "work_dir": str(run_dir),
            "plot": False,
            "save_all_checkpoints": False,
            # Avoid post-train deepcopy/torch.save; validation only needs log stats.
            "skip_save_model": True,
        }
    )

    # Ensure short smoke tests still produce at least one timing record.
    if config.get("cuda_timing", False):
        configured_warmup = int(config.get("cuda_timing_warmup", 0))
        config["cuda_timing_warmup"] = min(configured_warmup, epochs - 1)

    config_path = run_dir / "train_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path, name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def epoch_average_train_losses(
    train_records: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    """Average per-batch optimization losses into one train loss per epoch."""
    losses_by_epoch: dict[int, list[float]] = defaultdict(list)
    for record in train_records:
        if record.get("mode") != "opt" or record.get("epoch") is None:
            continue
        losses_by_epoch[int(record["epoch"])].append(float(record["loss"]))

    return [
        (epoch, statistics.fmean(losses))
        for epoch, losses in sorted(losses_by_epoch.items())
    ]


def epoch_valid_losses(
    train_records: list[dict[str, Any]],
) -> list[tuple[int, float]]:
    return [
        (int(record["epoch"]), float(record["loss"]))
        for record in train_records
        if record.get("mode") == "eval" and record.get("epoch") is not None
    ]


def write_loss_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss"])
        writer.writerows(rows)


def write_run_loss_csvs(
    loss_csv_dir: Path,
    run_index: int,
    train_records: list[dict[str, Any]],
) -> tuple[Path, Path]:
    train_rows = epoch_average_train_losses(train_records)
    valid_rows = epoch_valid_losses(train_records)
    if not train_rows:
        raise RuntimeError(f"No train losses found for run {run_index}")
    if not valid_rows:
        raise RuntimeError(f"No validation losses found for run {run_index}")

    train_csv = loss_csv_dir / f"train_loss_run_{run_index:02d}.csv"
    # Held-out validation curve; named test_loss for external comparison scripts.
    test_csv = loss_csv_dir / f"test_loss_run_{run_index:02d}.csv"
    write_loss_csv(train_csv, train_rows)
    write_loss_csv(test_csv, valid_rows)
    return train_csv, test_csv


def collect_run_stats(
    run_dir: Path, name: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tag = f"{name}_run-{seed}"
    train_records = read_jsonl(run_dir / "results" / f"{tag}_train.txt")
    timing_records = read_jsonl(run_dir / "results" / f"{tag}_cuda_timing.txt")

    eval_records = [
        record
        for record in train_records
        if record.get("mode") == "eval" and record.get("epoch") is not None
    ]
    if not eval_records:
        raise RuntimeError(f"No validation records found for {tag}")

    final_eval = max(eval_records, key=lambda record: int(record["epoch"]))
    best_eval = min(eval_records, key=lambda record: float(record["loss"]))
    final_timing = (
        max(timing_records, key=lambda record: int(record["epoch"]))
        if timing_records
        else {}
    )
    train_loss_curve = epoch_average_train_losses(train_records)
    if "train_loss" in final_timing:
        final_train_loss = float(final_timing["train_loss"])
    elif train_loss_curve:
        final_train_loss = train_loss_curve[-1][1]
    else:
        final_train_loss = None

    def metric(record: dict[str, Any], key: str, scale: float = 1.0):
        value = record.get(key)
        return None if value is None else float(value) * scale

    stats = {
        "seed": seed,
        "epochs_completed": int(final_eval["epoch"]) + 1,
        "final_train_loss": final_train_loss,
        "final_valid_loss": metric(final_eval, "loss"),
        "best_valid_loss": metric(best_eval, "loss"),
        "best_valid_epoch": int(best_eval["epoch"]),
        "final_energy_mae_per_atom_meV": metric(
            final_eval, "mae_e_per_atom", 1000.0
        ),
        "final_energy_rmse_per_atom_meV": metric(
            final_eval, "rmse_e_per_atom", 1000.0
        ),
        "final_force_mae_meV_per_A": metric(final_eval, "mae_f", 1000.0),
        "final_force_rmse_meV_per_A": metric(final_eval, "rmse_f", 1000.0),
        "mean_train_time_ms": mean_or_none(
            [float(record["train_time_ms"]) for record in timing_records]
        ),
        "mean_valid_time_ms": mean_or_none(
            [float(record["valid_time_ms"]) for record in timing_records]
        ),
        "peak_memory_mb": (
            max(float(record["peak_memory_mb"]) for record in timing_records)
            if timing_records
            else None
        ),
        "timed_epochs": len(timing_records),
    }
    return stats, train_records


def aggregate_stats(run_stats: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {"seed", "best_valid_epoch", "epochs_completed", "timed_epochs"}
    metric_names = sorted(
        {
            key
            for stats in run_stats
            for key, value in stats.items()
            if key not in excluded and isinstance(value, (int, float))
        }
    )

    aggregate: dict[str, Any] = {"num_runs": len(run_stats)}
    for name in metric_names:
        values = [
            float(stats[name])
            for stats in run_stats
            if isinstance(stats.get(name), (int, float))
        ]
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return aggregate


def print_stats(run_stats: list[dict[str, Any]], aggregate: dict[str, Any]) -> None:
    print("\nPer-run statistics")
    print(json.dumps(run_stats, indent=2))
    print("\nAggregate statistics (mean/std/min/max)")
    print(json.dumps(aggregate, indent=2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    train_path, valid_path, _ = create_split(args)
    base_config = load_base_config(args.base_config.resolve())
    loss_csv_dir = args.loss_csv_dir.resolve() if args.loss_csv_dir else None
    if loss_csv_dir is not None:
        loss_csv_dir.mkdir(parents=True, exist_ok=True)

    all_run_stats = []
    for run_index in range(args.num_runs):
        seed = args.first_run_seed + run_index
        run_dir = output_dir / "runs" / f"run_{run_index:02d}"
        config_path, name = write_run_config(
            base_config=base_config,
            run_dir=run_dir,
            train_path=train_path,
            valid_path=valid_path,
            molecule=args.molecule,
            run_index=run_index,
            seed=seed,
            epochs=args.epochs,
        )

        print(
            f"\n=== Run {run_index + 1}/{args.num_runs}: "
            f"seed={seed}, epochs={args.epochs} ==="
        )
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_train.py"),
                "--config",
                str(config_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        stats, train_records = collect_run_stats(run_dir, name, seed)
        if loss_csv_dir is not None:
            train_csv, test_csv = write_run_loss_csvs(
                loss_csv_dir, run_index, train_records
            )
            print(f"Wrote {train_csv}")
            print(f"Wrote {test_csv}")
        all_run_stats.append(stats)
        print(json.dumps(stats, indent=2))

    aggregate = aggregate_stats(all_run_stats)
    (output_dir / "run_stats.json").write_text(
        json.dumps(all_run_stats, indent=2) + "\n"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    print_stats(all_run_stats, aggregate)


if __name__ == "__main__":
    main()
