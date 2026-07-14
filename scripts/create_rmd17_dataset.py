#!/usr/bin/env python3
"""Download rMD17 via PyG and export train/valid splits for MACE training."""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import ase.io
import numpy as np
import yaml
from ase import Atoms
from ase.units import kcal, mol
from torch_geometric.datasets import MD17

KCAL_MOL_TO_EV = kcal / mol

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "rmd17"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "rmd17" / "pyg_cache"
DEFAULT_BASE_CONFIG = SCRIPT_DIR / "rmd17_train_base.yaml"

RMD17_MOLECULES = [
    "revised benzene",
    "revised uracil",
    "revised naphthalene",
    "revised aspirin",
    "revised salicylic acid",
    "revised malonaldehyde",
    "revised ethanol",
    "revised toluene",
    "revised paracetamol",
    "revised azobenzene",
]

# PyG's Materials Cloud URL is currently broken; Figshare hosts the same files.
FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/12672038"
FIGSHARE_TAR_URL = "https://ndownloader.figshare.com/files/23950376"


def pyg_data_to_atoms(data) -> Atoms:
    z = data.z.numpy()
    pos = data.pos.numpy()
    energy = float(data.energy.item()) * KCAL_MOL_TO_EV
    forces = data.force.numpy() * KCAL_MOL_TO_EV

    atoms = Atoms(numbers=z, positions=pos)
    atoms.center(vacuum=6.0)
    atoms.info["REF_energy"] = energy
    atoms.arrays["REF_forces"] = forces
    atoms.info["config_type"] = "Default"
    return atoms


def _figshare_npz_urls() -> dict[str, str]:
    with urllib.request.urlopen(FIGSHARE_ARTICLE_API) as response:
        article = json.load(response)
    return {
        file_info["name"]: file_info["download_url"]
        for file_info in article["files"]
        if file_info["name"].endswith(".npz")
    }


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} from Figshare...")
    urllib.request.urlretrieve(url, destination)


def ensure_rmd17_raw_data(cache_dir: Path, molecule: str) -> None:
    """Populate PyG's raw directory when the upstream Materials Cloud URL is unavailable."""
    npz_name = MD17.file_names[molecule]
    raw_npz_path = cache_dir / "raw" / "rmd17" / "npz_data" / npz_name
    if raw_npz_path.exists():
        return

    figshare_urls = _figshare_npz_urls()
    if npz_name in figshare_urls:
        _download_file(figshare_urls[npz_name], raw_npz_path)
        return

    tar_path = cache_dir / "raw" / "rmd17.tar.bz2"
    if not tar_path.exists():
        _download_file(FIGSHARE_TAR_URL, tar_path)

    print("Extracting rMD17 archive...")
    with tarfile.open(tar_path, mode="r:bz2") as archive:
        archive.extractall(cache_dir / "raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rMD17 via PyG and export MACE-compatible train/valid splits."
    )
    parser.add_argument(
        "--molecule",
        default="revised aspirin",
        choices=RMD17_MOLECULES,
        help="rMD17 molecule to download (default: revised aspirin)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory where PyG caches the downloaded dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/valid xyz files are written",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=950,
        help="Number of training configurations (default: 950)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=50,
        help="Number of validation configurations (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling before splitting (default: 0)",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Base MACE training YAML config to extend with dataset paths",
    )
    return parser.parse_args()


def write_train_config(
    base_config_path: Path,
    output_path: Path,
    train_path: Path,
    valid_path: Path,
    molecule: str,
    seed: int,
) -> None:
    with base_config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    mol_slug = molecule.replace(" ", "_")
    config["name"] = f"rmd17_{mol_slug}"
    config["seed"] = seed
    config["train_file"] = str(train_path.relative_to(REPO_ROOT))
    config["valid_file"] = str(valid_path.relative_to(REPO_ROOT))

    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    total = args.train_size + args.valid_size

    print(f"Preparing rMD17 '{args.molecule}' for PyG...")
    ensure_rmd17_raw_data(args.cache_dir, args.molecule)

    print("Loading dataset via PyG...")
    dataset = MD17(root=str(args.cache_dir), name=args.molecule)
    print(f"Loaded {len(dataset)} configurations")

    if len(dataset) < total:
        raise ValueError(
            f"Dataset has {len(dataset)} configurations, but "
            f"{total} are required ({args.train_size} train + {args.valid_size} valid)"
        )

    indices = np.arange(total)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)

    train_idx = indices[: args.train_size]
    valid_idx = indices[args.train_size :]

    train_atoms = [pyg_data_to_atoms(dataset[i]) for i in train_idx]
    valid_atoms = [pyg_data_to_atoms(dataset[i]) for i in valid_idx]

    mol_slug = args.molecule.replace(" ", "_")
    output_dir = args.output_dir / mol_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.xyz"
    valid_path = output_dir / "valid.xyz"
    config_path = output_dir / "train_config.yaml"
    manifest_path = output_dir / "split.json"

    ase.io.write(train_path, train_atoms, format="extxyz")
    ase.io.write(valid_path, valid_atoms, format="extxyz")
    write_train_config(
        args.base_config,
        config_path,
        train_path,
        valid_path,
        args.molecule,
        args.seed,
    )

    manifest = {
        "molecule": args.molecule,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "seed": args.seed,
        "train_indices": train_idx.tolist(),
        "valid_indices": valid_idx.tolist(),
        "train_file": str(train_path),
        "valid_file": str(valid_path),
        "train_config": str(config_path),
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "energy_units": "eV",
        "forces_units": "eV/Angstrom",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {len(train_atoms)} training configs to {train_path}")
    print(f"Wrote {len(valid_atoms)} validation configs to {valid_path}")
    print(f"MACE training config written to {config_path}")
    print(f"Split manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
