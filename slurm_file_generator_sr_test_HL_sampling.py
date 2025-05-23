import subprocess


def get_slurm_script(k):
    return f'''#!/bin/bash
#SBATCH --job-name=DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}_seed_%a # job name
#SBATCH --time=02:00:00   # maximum execution time (HH:MM:SS)
#SBATCH --output=YOUR_FOLDER/slurm_logs/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}_seed_%a.out # output
#SBATCH --error=YOUR_FOLDER/slurm_logs/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}_seed_%a.err # err
#SBATCH --partition=gpu_p6
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0,2,3


module purge
module load arch/h100
module load python/3.9.12
conda activate DLP

chmod +x slurm/launcher.sh
export "DLP_STORAGE"=YOUR_FOLDER/DLP

MASTER_PORT=$((34567+SLURM_ARRAY_TASK_ID))

srun slurm/launcher.sh \
                    rl_script_args.path=YOUR_FOLDER/DLP/test_hierarchical_agent.py \
                    test_training_script.epoch_k={k} \
                    test_training_script.nbr_tests_hierarchical_agent=50 \
                    test_training_script.goal_space=["go_to_tree","collect_wood","place_table","make_wood_pickaxe","go_to_stone","collect_stone"] \
                    test_training_script.test_name=test_HL_sampling \
                    rl_script_args.seed=${{SLURM_ARRAY_TASK_ID}} \
                    rl_script_args.number_envs=48 \
                    rl_script_args.epochs=520 \
                    rl_script_args.steps_per_epoch=2496 \
                    rl_script_args.max_tokens_per_epoch=9000 \
                    rl_script_args.hl_only_useful_tokens=false \
                    rl_script_args.add_eos_on_candidates=true \
                    rl_script_args.output_dir=YOUR_FOLDER \
                    hydra.run.dir=YOUR_FOLDER/hydra_file/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}/seed_${{SLURM_ARRAY_TASK_ID}} \
                    rl_script_args.name_experiment=DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5 \
                    rl_script_args.minibatch_size=3072 \
                    rl_script_args.gradient_batch_size=4 \
                    rl_script_args.entropy_coef=1e-2 \
                    rl_script_args.lr=1e-5 \
                    rl_script_args.lam=0.9 \
                    rl_script_args.gamma=0.95 \
                    rl_script_args.goal_sampler=MAGELLANGoalSampler \
                    rl_script_args.task_space=["go_to_tree","collect_wood","place_table","go_to_table","make_wood_pickaxe","go_to_stone","collect_stone","go_to_coal","collect_coal","place_furnace","go_to_furnace"] \
                    rl_script_args.long_description=true \
                    rl_script_args.actor_critic_separated=true \
                    rl_script_args.infinite_LL_traj=true \
                    rl_script_args.memory_ll=false \
                    rl_script_args.hl_traj_len_max=64 \
                    rl_script_args.ll_traj_len_max=128 \
                    rl_script_args.env_max_step=155 \
                    rl_script_args.reset_word_only_at_episode_termination=true \
                    rl_script_args.compute_kl_with_original_model=true \
                    rl_script_args.kl_penalty_target_low_level=0.01 \
                    rl_script_args.max_size_awr_buffer=100000 \
                    rl_script_args.minibatch_size_awr_low_level=256 \
                    rl_script_args.entro_coef_awr=0 \
                    rl_script_args.normalisation_coef_awr=true \
                    rl_script_args.lr_low_level_actor=1e-4 \
                    rl_script_args.lr_low_level_critic=1e-3 \
                    rl_script_args.use_lora=true \
                    rl_script_args.nn_approximation.explo_noise=0.1 \
                    rl_script_args.nn_approximation.update_ll_sr_estimator_after_n_transitions=256 \
                    rl_script_args.nn_approximation.batch_size_ll_sr_estimator=256 \
                    rl_script_args.nn_approximation.epochs_ll_sr_estimator=2 \
                    rl_script_args.nn_approximation.lr_ll_sr_estimator=1e-4 \
                    lamorel_args.llm_configs.main_llm.handler=unsloth \
                    lamorel_args.llm_configs.main_llm.model_type=causal \
                    lamorel_args.llm_configs.main_llm.model_path=YOUR_FOLDER/mistral-7b-v0.3-bnb-4bit \
                    lamorel_args.llm_configs.main_llm.pretrained=true \
                    lamorel_args.llm_configs.main_llm.load_in_4bit=true \
                    lamorel_args.llm_configs.main_llm.minibatch_size=240 \
                    lamorel_args.allow_subgraph_use_whith_gradient=true \
                    lamorel_args.distributed_setup_args.n_rl_processes=1 \
                    lamorel_args.distributed_setup_args.llm_processes.main_llm.n_processes=4 \
                    lamorel_args.distributed_setup_args.llm_processes.main_llm.devices_per_process=[[0],[0],[0],[0]] \
                    --config-path=YOUR_FOLDER/DLP \
                    --config-name=local_gpu_config
'''

n_runs = 0
for k in range(17, 21):
    with open(f'YOUR_FOLDER/DLP/test_sr/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}.slurm', 'w') as f:
        f.write(get_slurm_script(k))
    subprocess.run(['sbatch', f'YOUR_FOLDER/DLP/test_sr/DLP_textcrafter_gtt_to_pf_unsloth_ll_sr_update_256_lr_HL_1e5_test_HL_sampling_epoch_{k}.slurm'])
    n_runs += 1


print("Total Pareto front runs: " + str(n_runs))