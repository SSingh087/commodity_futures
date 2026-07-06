#!/usr/bin/env python3
"""
Generate the HTCondor DAG for parallel options data generation + merge.

Usage (run from project root):
    python gen_DAG.py --n_chunks 5 --config ./config/SchwartzSmithFWD.yaml
"""
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--n_chunks", type=int, default=500)
parser.add_argument("--config",   default="./config/SchwartzSmithFWD.yaml")
args = parser.parse_args()

N   = args.n_chunks
# Absolute paths so DAGMan can find files regardless of its working directory
ROOT       = os.path.abspath(".")
CFG        = os.path.abspath(args.config)
WORKER_SUB = os.path.join(ROOT, "SUBMIT_FILES", "generate_options_data.sub")
MERGE_SUB  = os.path.join(ROOT, "SUBMIT_FILES", "merge_options_data.sub")

lines = []

# ── Worker jobs ────────────────────────────────────────────────────────────
for i in range(N):
    job = f"OPTIONS_chunk{i:04d}"
    lines += [
        f"JOB      {job}  {WORKER_SUB}",
        # Quote values so strict-mode doesn't flag them as unused macros
        f'VARS     {job}  CHUNK_ID="{i}" N_CHUNKS="{N}" CONFIG="{CFG}"',
        f"CATEGORY {job}  options_workers",
        "",
    ]

# ── Cap concurrent jobs ────────────────────────────────────────────────────
lines.append("MAXJOBS options_workers 50")
lines.append("")

# ── Merge job ──────────────────────────────────────────────────────────────
merge_job = "MERGE_options"
lines += [
    f"JOB  {merge_job}  {MERGE_SUB}",
    f'VARS {merge_job}  N_CHUNKS="{N}" CONFIG="{CFG}"',
    "",
]

chunk_jobs = " ".join(f"OPTIONS_chunk{i:04d}" for i in range(N))
lines.append(f"PARENT {chunk_jobs} CHILD {merge_job}")

dag_path = f"dag_options_{N}chunks.dag"
with open(dag_path, "w") as f:
    f.write("\n".join(lines))

print(f"Written: {dag_path}  ({N} worker jobs + 1 merge job)")
print(f"Submit files resolved to: {WORKER_SUB}")