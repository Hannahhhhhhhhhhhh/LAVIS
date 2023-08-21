import sys
sys.path.append("/workspace/code/LAVIS/")
from lavis.tasks.captioning import CaptionTask
result_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/Caption_coco_flant5xxl_eval_0808/20230809140/result/test_epochbest.json"
coco_cn_caption_tast = CaptionTask(        
    num_beams=5,
    max_len=30,
    min_len=1,
    evaluate=True
    )
coco_cn_caption_tast._report_metrics(result_file, "test")