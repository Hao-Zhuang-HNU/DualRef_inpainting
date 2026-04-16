import os
import collections
import torch
from torch.utils.data import Dataset

from .dataset_TSR import ContinuousEdgeLineDatasetMaskFinetune


class ContinuousEdgeLineDatasetMaskFinetuneDualRef(Dataset):
    """Sequence-aware wrapper for TSR finetuning.

    Current frame uses the original finetune sample.
    Global reference = first frame in the same sequence.
    Local reference = previous frame in the same sequence.
    """

    def __init__(self, pt_dataset, mask_path=None, test_mask_path=None,
                 is_train=False, mask_rates=None, image_size=256, line_path=None,
                 no_global_ref=False, no_local_ref=False):
        self.dataset = ContinuousEdgeLineDatasetMaskFinetune(
            pt_dataset=pt_dataset,
            mask_path=mask_path,
            test_mask_path=test_mask_path,
            is_train=is_train,
            mask_rates=mask_rates,
            image_size=image_size,
            line_path=line_path,
        )
        self.image_id_list = self.dataset.image_id_list
        self.no_global_ref = no_global_ref
        self.no_local_ref = no_local_ref
        self.idx_info = {}

        seq_to_indices = collections.defaultdict(list)
        for i, path in enumerate(self.image_id_list):
            seq_to_indices[os.path.dirname(path)].append(i)

        for seq_id, idxs in seq_to_indices.items():
            sorted_pairs = sorted([(self.image_id_list[i], i) for i in idxs], key=lambda x: x[0])
            ordered = [i for _, i in sorted_pairs]
            global_idx = ordered[0]
            for j, curr_idx in enumerate(ordered):
                self.idx_info[curr_idx] = {
                    'seq_id': seq_id,
                    'global_idx': global_idx,
                    'prev_idx': ordered[j - 1] if j > 0 else -1,
                    'is_first': j == 0,
                }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        curr = self.dataset[idx]
        info = self.idx_info[idx]

        if self.no_global_ref:
            g_img = torch.zeros_like(curr['img'])
            g_line = torch.zeros_like(curr['line'])
        else:
            g_item = self.dataset[info['global_idx']]
            g_img = g_item['img']
            g_line = g_item['line']

        if self.no_local_ref or info['is_first']:
            l_img = torch.zeros_like(curr['img'])
            l_line = torch.zeros_like(curr['line'])
            l_mask = torch.ones_like(curr['mask'])
        else:
            l_item = self.dataset[info['prev_idx']]
            l_img = l_item['img']
            l_line = l_item['line']
            l_mask = torch.zeros_like(curr['mask'])

        out = dict(curr)
        out.update({
            'g_img': g_img.contiguous(),
            'g_line': g_line.contiguous(),
            'l_img': l_img.contiguous(),
            'l_line': l_line.contiguous(),
            'l_mask': l_mask.contiguous(),
            'is_first': torch.tensor(1.0 if info['is_first'] else 0.0, dtype=torch.float32),
            'orig_idx': idx,
        })
        return out
