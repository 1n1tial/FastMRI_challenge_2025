import h5py
import torch
from utils.data.transforms import DataTransform
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from utils.model.mraugment.data_augment import DataAugmentor
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from utils.model.mraugment.data_transforms import VarNetDataTransform

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, args, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.args = args
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        
        dataset_type = args.dataset_type.lower()  # "knee" or "brain"
        assert args.dataset_type.lower() in {"knee", "brain"}, "dataset_type must be 'knee' or 'brain'"
        
        if not forward:
            all_image_files = list(Path(root / "image").iterdir())
            image_files = [
                f for f in all_image_files if dataset_type in f.name.lower()
            ]
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]

        all_kspace_files = list(Path(root / "kspace").iterdir())
        kspace_files = [
            f for f in all_kspace_files if dataset_type in f.name.lower()
        ]
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        return self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)


def create_data_loaders(data_path, args, shuffle=False, isforward=False, data_augmentor=None):
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    if not isforward:
        if data_augmentor != None:
            transform = VarNetDataTransform(augmentor=data_augmentor, mask_func=mask, use_seed=False)
        else:
            transform = VarNetDataTransform(mask_func=mask)
    else:
        transform = VarNetDataTransform()
    if not isforward:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1
    data_storage = SliceData(
        root=data_path,
        transform=transform,
        input_key=args.input_key,
        target_key=target_key_,
        args=args,
        forward = isforward,
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
    )
    return data_loader
