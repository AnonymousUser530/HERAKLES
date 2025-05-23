#!/bin/bash

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n "1p") # Get IP (or hostname) of the first machine used
echo "running process ${SLURM_PROCID} on node $(hostname) with master ${MASTER_ADDR}"
python -m lamorel_launcher.launch lamorel_args.distributed_setup_args.multinode_args.main_process_ip=$MASTER_ADDR $*
