#!/bin/sh

python gen_dag.py \
    --stage2_root /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD_OOF \
    --output_root /data/wiay/postgrads/shashwat/COMM_DATA/results/CALLIBRATION_S3_L2 \
    --plots_root  /data/wiay/postgrads/shashwat/COMM_DATA/results/plots/CALLIBRATION_S3_L2 \
    --inj_sub     SUBMIT_FILES/inj_inf.sub \
    --ppplot_sub  SUBMIT_FILES/pp_plot.sub \
    --dag_path    SchwartzSmithFWD_OOF.dag \
    --n_injections 70 \
    --n_contracts 10