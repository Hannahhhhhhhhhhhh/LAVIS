GPUS_PER_NODE=4
WORKER_CNT=1
MASTER_PORT=25645
export CUDA_VISIBLE_DEVICES=1,4,6,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate.py --cfg-path lavis/projects/blip2/eval/gqa_zeroshot_flant5xxl_eval.yaml