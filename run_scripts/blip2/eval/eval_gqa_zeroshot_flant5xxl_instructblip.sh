GPUS_PER_NODE=2
WORKER_CNT=2
MASTER_PORT=25645
export CUDA_VISIBLE_DEVICES=2,7
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate.py --cfg-path lavis/projects/blip2/eval/gqa_zeroshot_flant5xxl_instructblip_eval.yaml