"""
Convert the raw Lending Club accepted-loans CSV into chunked Parquet files.

Why this exists (Phase 1 rationale):
- The raw CSV is ~1.6GB / 2.26M rows / 151 columns and will not fit in memory on a
  modest machine, so we stream it in chunks rather than loading it whole.
- We deliberately load every column as dtype=str (not pandas' inferred types).
  We haven't yet decided what each column's real type should be, or whether it's
  even usable (leakage, mostly-missing, etc) -- that's an explicit Phase 1 analysis
  step, not something to let a CSV parser silently do.
- Missing values still come through as true nulls (NaN), not the string "nan" --
  pandas' NA detection runs before the str cast.
- SCHEMA FIX (found the hard way): some columns -- bureau/joint-applicant fields
  Lending Club only started collecting partway through 2007-2018, e.g. open_acc_6m,
  il_util, the sec_app_* fields -- are 100% null in some chunks and populated in
  others, because the raw CSV isn't sorted by issue_d. Even with dtype=str, pyarrow's
  to_parquet will infer an all-null chunk's column as parquet type `null` rather than
  `string`, so different chunk files end up with different schemas and concatenation
  breaks. Fixed by reading the header once to fix the column list, building an
  explicit all-string pyarrow schema up front, and casting every chunk to it before
  writing -- regardless of whether that chunk happened to be all-null for some column.
- Output is one Parquet file per chunk rather than one giant file, so a crash
  partway through doesn't lose completed work and so no single step needs more
  memory than one chunk.
"""
import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Source CSV: "accepted_2007_to_2018Q4.csv" from the Kaggle dataset
# wordsforthewise/lending-club. Place it at data/raw/ or point LC_RAW_CSV at it.
SRC = os.environ.get("LC_RAW_CSV", "data/raw/accepted_2007_to_2018Q4.csv")
OUT_DIR = "data/raw/parquet"
CHUNKSIZE = 50_000

if not os.path.exists(SRC):
    raise SystemExit(
        f"Source CSV not found at: {SRC}\n\n"
        "Download 'accepted_2007_to_2018Q4.csv' from the Kaggle dataset\n"
        "  https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
        "and place it at data/raw/, or point LC_RAW_CSV at wherever you keep it:\n"
        "  LC_RAW_CSV=/path/to/accepted_2007_to_2018Q4.csv python -m data.convert_raw"
    )

os.makedirs(OUT_DIR, exist_ok=True)

header = pd.read_csv(SRC, nrows=0)
SCHEMA = pa.schema([pa.field(c, pa.string()) for c in header.columns])

t0 = time.time()
n_total = 0
n_chunks = 0

for i, chunk in enumerate(pd.read_csv(SRC, dtype=str, chunksize=CHUNKSIZE, low_memory=False)):
    table = pa.Table.from_pandas(chunk, schema=SCHEMA, preserve_index=False)
    out_path = os.path.join(OUT_DIR, f"part_{i:04d}.parquet")
    pq.write_table(table, out_path)
    n_total += len(chunk)
    n_chunks += 1
    print(
        f"chunk {i:04d}: {len(chunk)} rows -> {out_path} "
        f"(total {n_total} rows, {time.time() - t0:.1f}s elapsed)",
        flush=True,
    )

print(f"DONE. total_rows={n_total} total_chunks={n_chunks} elapsed={time.time() - t0:.1f}s", flush=True)
