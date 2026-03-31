# Global Variables
NUM_CPU_CORE = 4
RANDOM_SEED = 1993
NB_RUN = 1
# --dataset options: cifar_10, cifar_100
# --tr_frame options: plain
# --tr_scene options: cil, til
# Other override options:
# 	--override_should_train_single_epoch \
# 	--override_moe_should_not_earlystop \
# 	--override_add_noise_for_testing \

# Minimal test for system runnability
min_test:
	CUDA_VISIBLE_DEVICES=0 python "main/pipe_plain.py" \
	--dataset cifar_100 \
	--tr_scene cil \
	--nb_runs 1 \
	--nb_experts 1 \
	--nb_rounds 10 \
	--override_should_not_load_task0_ckp \
	--case_study "none" \
	--notes "minimal test" \
	--num_workers $(NUM_CPU_CORE) \
	--random_seed $(RANDOM_SEED)

run_cifar100:
	CUDA_VISIBLE_DEVICES=0 python "main/pipe_plain.py" \
	--dataset cifar_100 \
	--tr_scene cil \
	--nb_runs $(NB_RUN) \
	--nb_rounds 50 \
	--nb_experts 8 \
	--xt_scheme wf \
	--override_should_not_load_task0_ckp \
	--case_study "none" \
	--notes "M=4 on cifar-100" \
	--num_workers $(NUM_CPU_CORE) \
	--random_seed $(RANDOM_SEED)

.PHONY: min_test run_cifar100

