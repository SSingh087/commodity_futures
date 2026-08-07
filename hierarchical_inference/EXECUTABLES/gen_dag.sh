python gen_dag.py \
        --checkpoint_dir /data/wiay/postgrads/shashwat/COMM_DATA/results/checkpoints/SchwartzSmithFWD/bs2048_a0.20_b0.20_ep600_pat40_lr3.0e-04_wd1.0e-05_319402-2\
        --injection_dir  /data/wiay/postgrads/shashwat/COMM_DATA/results/population_results/injection \
        --output_root    /data/wiay/postgrads/shashwat/COMM_DATA/results/population_results/layer1_per_day \
        --hier_output    /data/wiay/postgrads/shashwat/COMM_DATA/results/population_results/layer2_hierarchical \
        --day_sub  /home/2673888s/commodity_futures/hierarchical_inference/SUBMIT_FILES/run_day_inf.sub \
        --hier_sub /home/2673888s/commodity_futures/hierarchical_inference/SUBMIT_FILES/run_pop_inf.sub \
        --dag_path /home/2673888s/commodity_futures/hierarchical_inference/population_inf.dag
