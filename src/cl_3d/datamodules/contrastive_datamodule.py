from typing import Optional, Tuple, Dict, Any

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

import hydra
import albumentations as A
import cv2

import pli_transforms.augmentations as aug
from pli_transforms.augmentations.pytorch import ToTensorPLI

from cl_3d import utils
from cl_3d.datamodules.components.data import TensorCollection


log = utils.get_logger(__name__)


class ContrastiveDataset(Dataset):

    def __init__(
            self,
            tensor_collection: TensorCollection,
            anchor_trans: Any = None,
            positive_trans: Any = None,
    ):
        super().__init__()

        self.tensor_collection = tensor_collection
        self.anchor_trans = anchor_trans
        self.positive_trans = positive_trans

    def __getitem__(self, locs):
        """
        :param loc: Locations of (anchor, positive) yielded by a PairSampler or PairBatchSampler
        :return: anchor_patch, positive_patch, anchor_location, positive_location
        """

        # Get pair indices
        anchor_location, positive_location = locs

        # Get image crops
        anchor_patch = self.tensor_collection[anchor_location]
        positive_patch = self.tensor_collection[positive_location]

        # Perform image transforms
        anchor_patch = self.anchor_trans(pli_dict=anchor_patch)['pli_dict']
        positive_patch = self.positive_trans(pli_dict=positive_patch)['pli_dict']

        return anchor_patch, positive_patch, anchor_location, positive_location

    def __len__(self):
        return 0


class ContrastiveDataModule(LightningDataModule):

    def __init__(
            self,
            tensor_collection: Dict,
            train_sampler: Dict,
            val_sampler: Dict,
            batch_size: int,
            patch_size: Tuple[int, int],
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.tensor_collection = hydra.utils.instantiate(tensor_collection)
        self.train_sampler = hydra.utils.instantiate(train_sampler)
        self.val_sampler = hydra.utils.instantiate(val_sampler)

        # data transformations
        self.transforms = A.Compose([
            aug.CleanPLI(),
            aug.AffinePLI(
                always_apply=True,
                scale={"x": (0.9, 1.3), "y": (0.9, 1.3)},
                rotate=(-180, 180),
                shear={"x": (-20, 20), "y": (-20, 20)},
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_REFLECT,
            ),
            aug.CenterCropPLI(
                height=self.hparams.patch_size[0],
                width=self.hparams.patch_size[1]
            ),
            aug.RandomDirectionOffsetPLI(),
            aug.RandomFlipPLI(),
            aug.ScaleThicknessPLI(
                log_range=(-1., 1.),
                trans_max=1.0,
                clip_max=1.5,
                always_apply=True
            ),
            aug.ScaleAttenuationPLI(
                log_range=(-1., 1.),
                trans_max=1.0,
                clip_max=1.5,
                always_apply=True
            ),
            aug.BlurPLI(
                blur_limit=(3, 3),
                sigma_limit=(0, 2.0),
                p=0.5
            ),
            ToTensorPLI()
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if stage in ['fit', None]:
            if not self.train_dataset or not self.val_dataset:
                self.tensor_collection.setup()
                if self.train_sampler:
                    self.train_sampler.setup()
                if self.val_sampler:
                    self.val_sampler.setup()

                self.train_dataset = ContrastiveDataset(
                    self.tensor_collection,
                    self.transforms,
                    self.transforms,
                )

                self.val_dataset = ContrastiveDataset(
                    self.tensor_collection,
                    self.transforms,
                    self.transforms,
                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            sampler=self.train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            sampler=self.val_sampler
        )
