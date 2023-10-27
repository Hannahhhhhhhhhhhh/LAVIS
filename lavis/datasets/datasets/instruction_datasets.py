"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "conversations": ' '.join([f"{item['from']}: {item['value']}" for item in ann["conversations"]]),
                "image": sample["image"],
            }
        )


class LlavaInstructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        conversations = [
            {
                'from': item['from'],
                'value': self.text_processor(item['value'])
            } for item in ann['conversations']
        ]

        return {"image": image, "conversations": conversations}
    
    def collater(self, samples):
        images = []
        conversations = []
        for sample in samples:
            images.append(sample['image'])
            conversations.append(sample['conversations'])
        
        return {
            "image": torch.stack(images),
            "conversations": conversations
        }


class LlavaLiInstructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        text_input = ann['text_input']
        text_output = ann['text_output']

        return {"image": image, "text_input": text_input, "text_output": text_output}


class LlavaSftInstructionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        text_input = ann['text_input']
        text_output = ann['text_output']

        return {"image": image, "text_input": text_input, "text_output": text_output, "image_id": ann['image']}


class LlavaSftInstructionEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        if "COCO" in ann["image"]: # coco images
            image_path = os.path.join(self.vis_root, ann["image"])
            text_input = ann['text_input']
            text_output = ann['text_output']
        else: # gqa images
            image_path = os.path.join("/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/images/", ann["image"])
            text_input = ann["question"]
            text_output =  ann["fullAnswer"]

        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        return {
            "image": image,
            "image_id": ann["image"],
            "text_input": text_input,
            "answer": ann.get("answer",""),
            "fullAnswer": ann.get("fullAnswer",""),
            "text_output": text_output,
            "question_id": ann.get("question_id",""),
            "instance_id": ann.get("instance_id",""),
        }