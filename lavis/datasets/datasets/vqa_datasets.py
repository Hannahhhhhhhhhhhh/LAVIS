"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch

from lavis.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        image_list, question_list, answer_list, weight_list, text_output_list, image_id_list = [], [], [], [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

            if "text_output" in sample.keys():
                text_output_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "image_id": image_id_list,
            "text_input": question_list,
            "answer": answer_list,
            "weight": torch.Tensor(weight_list),
            "n_answers": torch.LongTensor(num_answers),
            "text_output": text_output_list,
        }


class VQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
