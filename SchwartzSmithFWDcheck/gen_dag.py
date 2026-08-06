"""
generate_stage3_dag.py — writes ONLY the .dag file.

Assumes inj_stage3.sub and ppplot_stage3.sub already exist (edit the
PYTHON_ENV / SCRIPT_DIR lines at the top of each once, per your setup).
This script just enumerates your trained Stage 1 models, writes one
JOB+VARS pair per (model, injection seed), and one PP-plot JOB+VARS per
model with a PARENT/CHILD line tying its injections to it.

Usage:
    python generate_stage3_dag.py \\
        --stage1_root /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD \\
        --output_root /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L1 \\
        --plots_root  /data/wiay/postgrads/shashwat/COMM_DATA/results/plots/CALLIBRATION_S3_L1 \\
        --inj_sub     /home/2673888s/commodity_futures/condor_stage3_layer1/inj_stage3.sub \\
        --ppplot_sub  /home/2673888s/commodity_futures/condor_stage3_layer1/ppplot_stage3.sub \\
        --dag_path    /home/2673888s/commodity_futures/condor_stage3_layer1/stage3_layer1.dag \\
        --n_injections 100

Then:
    cd /home/2673888s/commodity_futures/condor_stage3_layer1
    condor_submit_dag stage3_layer1.dag
"""

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize(name: str) -> str:
    """DAG node names shouldn't contain dots/plus/minus in some HTCondor
    versions — swap them out for underscore-safe equivalents."""
    return name.replace(".", "p").replace("+", "plus").replace("-", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_root", required=True,
                     help="Dir containing one subfolder per trained Stage 1 model")
    ap.add_argument("--output_root", required=True,
                     help="Where run_stage3_layer1.py writes inj_* results")
    ap.add_argument("--plots_root", required=True,
                     help="Where PP plots get saved (one per model)")
    ap.add_argument("--inj_sub", required=True, help="Path to inj_stage3.sub")
    ap.add_argument("--ppplot_sub", required=True, help="Path to ppplot_stage3.sub")
    ap.add_argument("--dag_path", required=True, help="Output path for the .dag file")
    ap.add_argument("--n_injections", type=int, default=100,
                     help="Injections per model. 100 is a reasonable default — "
                          "enough for meaningful KS-test p-values, cheap enough "
                          "to run in an afternoon. Bump it if a PP curve looks "
                          "borderline and you want tighter confidence.")
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    args = ap.parse_args()

    stage1_root = Path(args.stage1_root)
    model_dirs = sorted([p for p in stage1_root.iterdir() if p.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No model subfolders found under {stage1_root}")

    print(f"Found {len(model_dirs)} Stage 1 models:")
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
        dag_lines.append("")  # blank line between model blocks

    # Retry each injection job once — cheap insurance against a stray
    # eviction/OOM on a shared cluster over a campaign this size.
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
    print(f"\nSubmit with:\n  cd {dag_path.parent}\n  condor_submit_dag {dag_path.name}")


if __name__ == "__main__":
    main()