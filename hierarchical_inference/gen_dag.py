"""
gen_layer2_dag.py — writes the Layer 2 .dag file
======================================================

Assumes day_inf.sub / hier_inf.sub already exist (edit PYTHON_ENV/paths
at the top of each once). This only enumerates the days already saved by
gen_population_injection.py and writes the JOB/VARS/PARENT-CHILD lines.

Usage:
    # 1. Generate the injection first (draws Lambda_true + D days):
    python gen_population_injection.py \\
        --checkpoint_dir /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD/<best_model> \\
        --out_dir /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2_hier/injection \\
        --n_days 20

    # 2. Then generate the DAG:
    python gen_layer2_dag.py \\
        --checkpoint_dir /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD/<best_model> \\
        --injection_dir  /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2_hier/injection \\
        --output_root    /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2_hier/layer1_per_day \\
        --hier_output    /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2_hier/layer2_hierarchical \\
        --day_sub  /home/2673888s/commodity_futures/hierarchical_inference/SUBMIT_FILES/day_inf.sub \\
        --hier_sub /home/2673888s/commodity_futures/hierarchical_inference/SUBMIT_FILES/hier_inf.sub \\
        --dag_path /home/2673888s/commodity_futures/hierarchical_inference/layer2.dag

    # 3. Submit:
    cd /home/2673888s/commodity_futures/hierarchical_inference
    condor_submit_dag layer2.dag
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--injection_dir", required=True,
                     help="Output of gen_population_injection.py — contains "
                          "day_0000/, day_0001/, ..., lambda_true.npy")
    ap.add_argument("--output_root", required=True,
                     help="Where each day's Layer 1 posterior gets saved")
    ap.add_argument("--hier_output", required=True,
                     help="Where the final Lambda posterior gets saved")
    ap.add_argument("--day_sub", required=True, help="Path to day_inf.sub")
    ap.add_argument("--hier_sub", required=True, help="Path to hier_inf.sub")
    ap.add_argument("--dag_path", required=True)
    ap.add_argument("--n_live_day", type=int, default=1000)
    ap.add_argument("--n_live_hier", type=int, default=1000)
    ap.add_argument("--n_pool", type=int, default=4)
    args = ap.parse_args()

    injection_dir = Path(args.injection_dir)
    day_dirs = sorted(injection_dir.glob("day_*"))
    if not day_dirs:
        raise FileNotFoundError(f"No day_* subfolders found under {injection_dir} — "
                                 f"run gen_population_injection.py first.")
    lambda_true_path = injection_dir / "lambda_true.npy"
    if not lambda_true_path.exists():
        raise FileNotFoundError(f"{lambda_true_path} not found — "
                                 f"run gen_population_injection.py first.")

    print(f"Found {len(day_dirs)} days under {injection_dir}")

    dag_lines = []
    day_node_names = []

    for day_dir in day_dirs:
        day_tag = day_dir.name  # e.g. day_0007
        node_name = f"day_{day_tag}"
        day_node_names.append(node_name)
        dag_lines.append(f"JOB {node_name} {args.day_sub}")
        dag_lines.append(
            f'VARS {node_name} checkpoint_dir="{args.checkpoint_dir}" '
            f'day_dir="{day_dir}" output_root="{args.output_root}" '
            f'n_live="{args.n_live_day}" n_pool="{args.n_pool}"'
        )

    hier_node = "hierarchical"
    dag_lines.append(f"JOB {hier_node} {args.hier_sub}")
    dag_lines.append(
        f'VARS {hier_node} day_posteriors_root="{args.output_root}" '
        f'lambda_true_path="{lambda_true_path}" output_root="{args.hier_output}" '
        f'n_live="{args.n_live_hier}" n_pool="{args.n_pool}"'
    )
    dag_lines.append(f"PARENT {' '.join(day_node_names)} CHILD {hier_node}")

    for node_name in day_node_names:
        dag_lines.append(f"RETRY {node_name} 1")

    dag_path = Path(args.dag_path)
    dag_path.parent.mkdir(parents=True, exist_ok=True)
    dag_path.write_text("\n".join(dag_lines) + "\n")

    print(f"\nDAG written to {dag_path}")
    print(f"  {len(day_dirs)} day jobs + 1 hierarchical job = {len(day_dirs) + 1} total")
    print(f"\nSubmit with:\n  cd {dag_path.parent}\n  condor_submit_dag {dag_path.name}")


if __name__ == "__main__":
    main()