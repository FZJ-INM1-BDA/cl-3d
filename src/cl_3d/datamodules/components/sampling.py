from collections import namedtuple
from typing import Tuple, Optional

import os
import re
import math

import torch
from torch.utils.data.sampler import Sampler

import numpy as np
import pandas as pd
import SimpleITK as sitk

from cl_3d import utils


log = utils.get_logger(__name__)


Location = namedtuple('Location', ['dataset', 'row', 'col'])


def dfield_from_file(file):
    transform = sitk.DisplacementFieldTransform(sitk.ReadTransform(file))

    m = re.search('.*Vervet1818([abc]*)_.*_s([0-9]*)_.*', file)
    section_id = int(m.group(2))
    roi = m.group(1)
    return transform, roi, section_id


def warp_coord(coord, transform):
    return transform.TransformPoint(coord)


def coord2ix(coord, spacing):
    # Input spacing in mu
    return tuple(reversed([int(c // (s / 1_000)) for c, s in zip(coord, spacing)]))


class MyDistributedSampler(Sampler):

    def __init__(
            self,
            size: int,
            epoch_size: Optional[int] = None,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            shuffle_before_split: bool = True,
            seed: int = 0,
            drop_last: bool = True,
    ):
        """
        Distributed Sampler

        :param size: Size of the data as number of samples
        :param epoch_size: Size of samples drawn each epoch. This can differ from (max - min) of index_interval
        :param num_replicas: Number of processes to split the data. Leave as None to determine by MPI world size.
        :param rank: Local rank for the sampler. Leave as None to determine by MPI rank
        :param shuffle: If to shuffle the indices each epoch
        :param shuffle_before_split: If to shuffle the indices before or after split into num_replicas subsets. If set
        to false the subsets are kept constant for each replica over all epochs (e.g. for better caching). Keep in mind
        that data samples might not be i.i.d anymore by disabling this flag.
        :param seed: Initial seed for subsampling and transformation. It is updated using epoch and rank each epoch.
        :param drop_last: If to drop the last samples if the number of indices in epoch_size cannot be divided
        by num_replicas.
        """
        super().__init__(None)

        self.size = size
        self.epoch_size = epoch_size if epoch_size is not None else self.size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.shuffle_before_split = shuffle_before_split
        self.seed = seed
        self.drop_last = drop_last

        # Dummy entries. Should be overwritten in setup() call
        self.epoch = 0
        self.num_samples = self.epoch_size
        self.total_size = self.epoch_size

    def setup(self):
        # Has to be called in MPI context
        if self.num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.num_replicas = torch.distributed.get_world_size()
                log.info(f"Use {self.num_replicas} replications of the dataset")
            else:
                self.num_replicas = 1
        if self.rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.rank = torch.distributed.get_rank()
            else:
                self.rank = 0
        if self.rank >= self.num_replicas or self.rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(self.rank, self.num_replicas - 1))

        if self.drop_last:
            self.num_samples = math.floor(self.epoch_size / self.num_replicas)
        else:
            self.num_samples = math.ceil(self.epoch_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _sample_ix(self, g: torch.Generator):
        if self.shuffle and self.shuffle_before_split:
            # deterministically shuffle based on epoch and seed
            indices = torch.randperm(self.size, generator=g)
        else:
            indices = torch.arange(self.size)

        if self.total_size > len(indices):
            # add extra samples to fill epoch_size
            indices = indices.repeat(math.ceil(self.total_size / len(indices)))[:self.total_size]

        # subsample
        start_ix = int(self.rank * len(indices) // self.num_replicas)
        end_ix = int(start_ix + (len(indices) // self.num_replicas))
        indices = indices[start_ix:end_ix]

        # Reduce number of indices to num_samples
        if self.shuffle and not self.shuffle_before_split:
            indices = indices[torch.randperm(len(indices), generator=g)][:self.num_samples]
        else:
            indices = indices[:self.num_samples]

        assert len(indices) == self.num_samples

        return indices.tolist()

    def _yield_pair(self, ix: int, g: torch.Generator) -> Tuple[Location, Location]:
        pass

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(utils.better_seed([self.seed, self.epoch]).torch))
        samples = self._sample_ix(g)

        # DEBUG
        # print(f" {len(samples)} samples on {self.rank} at {self.epoch}:\n", samples)

        g.manual_seed(int(utils.better_seed([self.seed, self.epoch, self.rank]).torch))
        for ix in samples:
            yield self._yield_pair(ix, g)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. This method is called by lightning on dataloader preparation in function
        "loops.fit_loop.on_advance_start" but only for the training. For validation epoch is kept constant!

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedContextSampler(MyDistributedSampler):

    def __init__(
            self,
            samples_file: str,
            transforms_path: str,
            translate_mu: float = 128 * 1.3,
            thickness_mu: float = 60.,
            spacing_mu: Tuple[float, float] = (1.3, 1.3),
            thetas: Tuple[float, float] = (0, np.pi),
            phis: Tuple[float, float] = (0, 2 * np.pi),
            r_mu: float = 236.,
            h_mu: float = 236.,
            exclude_self: bool = True,
            index_interval: Optional[Tuple[int, int]] = None,
            epoch_size: Optional[int] = None,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            shuffle_before_split: bool = True,
            seed: int = 0,
            drop_last: bool = True,
    ):
        """
        Distributed Sampler for handling paired samples.

        :param samples_file: CSV file providing sampling locations of patches
        :param transforms_path: Path where transformations are stored. Will read everything that is inside the path.
        :param translate_mu: Random x and y offsets are drawn uniform from [-translate_mu, translate_mu]. Set to 0 to disable.
        :param thickness_mu: Section thickness as constant.
        :param spacing_mu: Section spacing as constant.
        :param thetas: Sampling interval for theta parameter of ellipsoid coordinates.
        :param phis: Sampling interval for phi paramter of ellipsoid coordinates.
        :param r_mu: Fixed in-plane radius for context.
        :param h_mu: Fixed out-pf-plane radius for context.
        :param exclude_self: Weather the same section is excluded from selection for the paired sample.
        :param index_interval: Interval [max, min] of indices used for the sampler
        :param epoch_size: Size of samples drawn each epoch. This can differ from (max - min) of index_interval
        :param num_replicas: Number of processes to split the data. Leave as None to determine by MPI world size.
        :param rank: Local rank for the sampler. Leave as None to determine by MPI rank
        :param shuffle: If to shuffle the indices each epoch
        :param shuffle_before_split: If to shuffle the indices before or after split into num_replicas subsets. If set
        to false the subsets are kept constant for each replica over all epochs (e.g. for better caching). Keep in mind
        that data samples might not be i.i.d anymore by disabling this flag.
        :param seed: Initial seed for subsampling and transformation. It is updated using epoch and rank each epoch.
        :param drop_last: If to drop the last samples if the number of indices in epoch_size cannot be divided
        by num_replicas.
        """

        # Read sampling points
        if index_interval is None:
            self.samples_df = pd.read_csv(samples_file, index_col=0).reset_index(drop=True)
        else:
            self.samples_df = pd.read_csv(samples_file, index_col=0).loc[np.arange(*index_interval)].reset_index(drop=True)
            
        super().__init__(
            size=len(self.samples_df),
            epoch_size=epoch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            shuffle_before_split=shuffle_before_split,
            seed=seed,
            drop_last=drop_last,
        )

        self.exclude_self = exclude_self
        self.h_mu = h_mu
        self.r_mu = r_mu
        self.phis = phis
        self.thetas = thetas
        self.spacing_mu = spacing_mu
        self.thickness_mu = thickness_mu
        self.translate_mm = translate_mu / 1_000

        # Get available datasets and their section number and roi
        self.dataset_df = self.samples_df[['dataset', 'section', 'roi']].drop_duplicates().reset_index(drop=True)
        self.dataset_df = self.dataset_df.sort_values(['section', 'roi'])

        # Load the transformation fields. The fields are stored with this sampler
        tf_list = os.listdir(transforms_path)
        self.tf_dict = {}
        for tf in tf_list:
            transform, roi, section_id = dfield_from_file(os.path.join(transforms_path, tf))
            self.tf_dict[(section_id, roi)] = transform

    def _yield_pair(self, ix: int, g: torch.Generator) -> Tuple[Location, Location]:

        # Lookup dataset, row and cl for pairs at given index
        sample = self.samples_df.loc[ix]

        # Get Sample position
        if self.translate_mm > 0:
            x = sample.x + 2 * self.translate_mm * torch.rand((1,), generator=g).item() - self.translate_mm
            y = sample.y + 2 * self.translate_mm * torch.rand((1,), generator=g).item() - self.translate_mm
        else:
            x = sample.x
            y = sample.y

        # Warp position for first section
        transform = self.tf_dict[(sample.section, sample.roi)]
        coord_warped = warp_coord((x, y), transform)
        row_1, col_1 = coord2ix(coord_warped, self.spacing_mu)

        # Get context offset
        theta = ((self.thetas[1] - self.thetas[0]) * torch.rand(1, generator=g) + self.thetas[0]).item()
        phi = ((self.phis[1] - self.phis[0]) * torch.rand(1, generator=g) + self.phis[0]).item()
        delta_p = (
            self.r_mu * np.sin(theta) * np.cos(phi),
            self.r_mu * np.sin(theta) * np.sin(phi),
            self.h_mu * np.cos(theta)
        )

        # Find closest dataset to delta_p[2] + sample.section
        z = (delta_p[2] / self.thickness_mu) + sample.section
        candidates = self.dataset_df.iloc[(self.dataset_df['section'] - z).abs().argsort()[:2]].index
        if self.exclude_self:
            if self.dataset_df.section[candidates[0]] == sample.section:
                pair_sample = self.dataset_df.loc[candidates[1]]
            else:
                pair_sample = self.dataset_df.loc[candidates[0]]
        else:
            pair_sample = self.dataset_df.loc[candidates[0]]

        # Warp position for second section_id
        x2 = x + (delta_p[0] / 1_000)  # mm
        y2 = y + (delta_p[1] / 1_000)  # mm
        transform = self.tf_dict[(pair_sample.section, pair_sample.roi)]
        coord_warped = warp_coord((x2, y2), transform)
        row_2, col_2 = coord2ix(coord_warped, self.spacing_mu)

        # Define paired output Locations
        location_pair = Location(sample.dataset, row_1, col_1), Location(pair_sample.dataset, row_2, col_2)

        return location_pair
