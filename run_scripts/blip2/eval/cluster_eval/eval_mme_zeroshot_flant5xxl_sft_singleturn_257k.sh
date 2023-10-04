GPUS_PER_NODE=1
WORKER_CNT=1
MASTER_PORT=25645
export CUDA_VISIBLE_DEVICES=7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate_cluster.py --cfg-path lavis/projects/blip2/eval/cluster_eval/mme_zeroshot_flant5xxl_sft_eval_default.yaml