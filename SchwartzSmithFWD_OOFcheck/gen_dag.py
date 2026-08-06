"""
gen_dag.py — Stage 3 Layer 2 (options) DAG generator
=========================================================

Same pattern as SchwartzSmithFWDcheck/gen_dag.py: writes ONLY the .dag
file, referencing the already-existing inj_inf.sub / pp_plot.sub in
SUBMIT_FILES/. One difference from Stage 1: injection jobs need an
extra n_contracts VAR (how many simulated option contracts make up one
synthetic "day").

Usage:
    python gen_dag.py \
        --stage2_root /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD_OOF \
        --output_root /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2 \
        --plots_root  /data/wiay/postgrads/shashwat/COMM_DATA/results/plots/CALLIBRATION_S3_L2 \
        --inj_sub     SUBMIT_FILES/inj_inf.sub \
        --ppplot_sub  SUBMIT_FILES/pp_plot.sub \
        --dag_path    SchwartzSmithFWD_OOF.dag \
        --n_injections 100 \
        --n_contracts 10

Then:
    condor_submit_dag SchwartzSmithFWD_OOF.dag
"""

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize(name: str) -> str:
    return name.replace(".", "p").replace("+", "plus").replace("-", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_root", required=True,
                     help="Dir containing one subfolder per trained Stage 2 "
                          "ensemble (e.g. .../checkpoints/SchwartzSmithFWD_OOF)")
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--plots_root", required=True)
    ap.add_argument("--inj_sub", required=True)
    ap.add_argument("--ppplot_sub", required=True)
    ap.add_argument("--dag_path", required=True)
    ap.add_argument("--n_injections", type=int, default=100)
    ap.add_argument("--n_contracts", type=int, default=10,
                     help="Simulated option contracts per synthetic day.")
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    args = ap.parse_args()

    stage2_root = Path(args.stage2_root)
    model_dirs = sorted([p for p in stage2_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No model subfolders found under {stage2_root}")

    print(f"Found {len(model_dirs)} Stage 2 models:")
    for m in model_dirs:
        print(f"  {m.name}")

    dag_lines = []

    for model_dir in model_dirs:
        model_tag = model_dir.name
        node_prefix = sanitize(model_tag)
        inj_node_names = []

        for seed in range(args.n_injections):
            node_name = f"inj_{node_prefix}_{seed:04d}"
            inj_node_names.append(node_name)
            dag_lines.append(f"JOB {node_name} {args.inj_sub}")
            dag_lines.append(
                f'VARS {node_name} checkpoint_dir="{model_dir}" '
                f'seed="{seed}" output_root="{args.output_root}" '
                f'n_contracts="{args.n_contracts}" '
                f'n_live="{args.n_live}" n_pool="{args.n_pool}"'
            )

        ppplot_node = f"ppplot_{node_prefix}"
        results_dir = Path(args.output_root) / model_tag
        out_path = Path(args.plots_root) / f"pp_plot_{model_tag}.png"
        dag_lines.append(f"JOB {ppplot_node} {args.ppplot_sub}")
        dag_lines.append(
            f'VARS {ppplot_node} results_dir="{results_dir}" out_path="{out_path}"'
        )
        dag_lines.append(f"PARENT {' '.join(inj_node_names)} CHILD {ppplot_node}")
        dag_lines.append("")

    for model_dir in model_dirs:
        node_prefix = sanitize(model_dir.name)
        for seed in range(args.n_injections):
            dag_lines.append(f"RETRY inj_{node_prefix}_{seed:04d} 1")

    dag_path = Path(args.dag_path)
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    dag_path.write_text("\n".join(dag_lines) + "\n")

    n_total = len(model_dirs) * (args.n_injections + 1)
    print(f"\nDAG written to {dag_path}")
    print(f"  {len(model_dirs)} models x ({args.n_injections} injections + 1 PP-plot) "
          f"= {n_total} total jobs")
    print(f"\nSubmit with:\n  condor_submit_dag {dag_path.name}")


if __name__ == "__main__":
    main()