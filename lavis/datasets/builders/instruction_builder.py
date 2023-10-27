
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.datasets.datasets.instruction_datasets import (
    LlavaInstructionDataset, LlavaSftInstructionDataset, LlavaSftInstructionEvalDataset
)
from lavis.datasets.datasets.gqa_datasets import GQAEvalDataset

@registry.register_builder("llava150k_en")
class LlavaEnBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaInstructionDataset
    eval_dataset_cls = LlavaInstructionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/llava/defaults_150k_en.yaml"}

@registry.register_builder("llava150k_en_sft")
class LlavaEnSftBuilder(BaseDatasetBuilder):
    train_dataset_cls = LlavaSftInstructionDataset
    eval_dataset_cls = LlavaSftInstructionEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/defaults_150k_en_sft.yaml",
        "prompt_moe": "configs/datasets/llava/defaults_150k_en_prompt_moe.yaml",
        "prompt_moe_test": "configs/datasets/llava/defaults_150k_en_prompt_moe_test.yaml",
        }

@registry.register_builder("llava_150k_en_sft_singleturn")
class LlavaEnSftBuilderSingleTurn(BaseDatasetBuilder):
    train_dataset_cls = LlavaSftInstructionDataset
    eval_dataset_cls = LlavaSftInstructionDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/llava/defaults_150k_en_sft_singleturn.yaml"}

@registry.register_builder("llava_150k_en_sft_singleturn_cluster")
class LlavaEnSftBuilderSingleTurnCluster(BaseDatasetBuilder):
    train_dataset_cls = LlavaSftInstructionDataset
    eval_dataset_cls = LlavaSftInstructionDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/llava/defaults_150k_en_sft_singleturn_cluster.yaml"
        }


