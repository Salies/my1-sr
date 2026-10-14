"""Build my1_srbench_datasets.tar.gz from the four MY1 CSV files.

Usage (any OS with Python 3 + pandas; run on your own computer):
    python make_my1_tarball.py <folder with the CSVs> [output.tar.gz]

Example:
    python make_my1_tarball.py C:\\Users\\me\\my1_data
    python make_my1_tarball.py ~/my1_data my1_srbench_datasets.tar.gz

Expects these files in the folder (a 'target' column plus numeric features):
    my1_daily_o3.csv  my1_daily_pm25.csv  my1_hourly_o3.csv  my1_hourly_pm25.csv

The archive extracts straight into srbench's datasets/ folder as:
    my1_daily/<name>/<name>.tsv.gz + metadata.yaml   (the two daily datasets)
    my1_hourly/<name>/<name>.tsv.gz + metadata.yaml  (the two hourly datasets)
"""
import io
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["my1_daily_o3", "my1_daily_pm25", "my1_hourly_o3", "my1_hourly_pm25"]
MAX_TRAIN = 40000  # SRBench's -max_samples; training set = 75% of rows


def add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def add_dir(tar: tarfile.TarFile, arcname: str) -> None:
    info = tarfile.TarInfo(arcname)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.mtime = int(time.time())
    tar.addfile(info)


def load_and_check(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    name = path.stem
    problems = []
    if "target" not in df.columns:
        problems.append("no 'target' column")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")
    if df.isna().any().any():
        problems.append(f"missing values in: {df.columns[df.isna().any()].tolist()}")
    numeric = df.select_dtypes("number")
    if np.isinf(numeric.to_numpy()).any():
        problems.append("infinite values present")
    if df.shape[1] < 2:
        problems.append("no feature columns")
    if problems:
        raise SystemExit(f"{name}: " + "; ".join(problems))

    # Warnings that don't stop the build
    dotted = [c for c in df.columns if "." in c or c != c.strip()]
    if dotted:
        print(f"  note: {name}: SRBench renames these columns ('.'->'_', spaces trimmed): {dotted}")
    constant = [c for c in df.columns if c != "target" and df[c].nunique() <= 1]
    if constant:
        print(f"  note: {name}: constant columns (carry no information): {constant}")
    if int(len(df) * 0.75) > MAX_TRAIN:
        print(f"  note: {name}: training set will be subsampled to {MAX_TRAIN} rows")

    # target last, like PMLB files
    return df[[c for c in df.columns if c != "target"] + ["target"]]


def main(src_dir: str, out_path: str) -> None:
    src = Path(src_dir).expanduser()
    missing = [n for n in NAMES if not (src / f"{n}.csv").exists()]
    if missing:
        raise SystemExit(f"Missing in {src}: " + ", ".join(f"{n}.csv" for n in missing))

    out = Path(out_path).expanduser()
    with tarfile.open(out, "w:gz") as tar:
        for group in ["my1_daily", "my1_hourly"]:
            add_dir(tar, group)
        for name in NAMES:
            df = load_and_check(src / f"{name}.csv")
            group = "my1_daily" if "_daily_" in name else "my1_hourly"
            folder = f"{group}/{name}"
            add_dir(tar, folder)

            buf = io.BytesIO()
            df.to_csv(buf, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
            add_bytes(tar, f"{folder}/{name}.tsv.gz", buf.getvalue())

            n_features = df.shape[1] - 1
            metadata = (
                f"dataset: {name}\n"
                f"description: MY1 (London Marylebone Road) air quality, target = {name.split('_')[-1]}\n"
                f"source: user-provided\n"
                f"task: regression\n"
                f"target:\n  type: continuous\n  description: concentration\n"
                f"n_instances: {len(df)}\n"
                f"n_features: {n_features}\n"
            )
            add_bytes(tar, f"{folder}/metadata.yaml", metadata.encode())
            print(f"{name}: {len(df)} rows, {n_features} features -> {folder}/")

    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "my1_srbench_datasets.tar.gz")