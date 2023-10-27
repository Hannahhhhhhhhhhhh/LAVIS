GPUS_PER_NODE=1
WORKER_CNT=1
MASTER_PORT=25647
export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} evaluate.py --cfg-path lavis/projects/blip2/eval/prompt_moe_eval/gqa_llava_prompt_flant5xxl_eval_1010.yaml