GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1236}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
PRECISION=${PRECISION:-bf16}


accelerate launch \
    --config-file accelerator/4gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --num_machines $NNODES \
    --mixed_precision $PRECISION \
    compute_fid_cifar.py --input_dir "results/icfm_gaussian" --start_step 80000 --end_step 180000 --step 10000 --num_gen 50000 --batch_size_fid 512 --model icfm --source gaussian