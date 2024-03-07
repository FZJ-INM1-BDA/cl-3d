from typing import Tuple, Any, Dict

import os

import torch
import numpy as np
import h5py as h5

from atlaslib.io import stage_split_hdf5
from atlaslib.files import require_copy, require_directory

from cl_3d import utils
from cl_3d.datamodules.components.sampling import Location

# Distributed
from atlasmpi import MPI


comm = MPI.COMM_WORLD


log = utils.get_logger(__name__)


class TensorCollection(object):

    def __init__(self):
        """Tensor collection of an unlimited size"""
        super().__init__()

    def setup(self):
        """Setups the data"""
        pass

    def __getitem__(self, loc: Any) -> Dict[str, torch.Tensor]:
        """Get a dictionary of named tensors at given location"""
        pass


class ModalityCollection(TensorCollection):

    def __init__(
            self,
            collection_file: str,
            crop_size: Tuple[int, int],
            to_ram: bool = False,
            driver: str = None,
            **h5kwargs
    ):
        super().__init__()

        # Open HDF5 file with optional additional kwargs
        self.collection_file = collection_file
        self.crop_size = crop_size
        self.crop_size = crop_size
        self.to_ram = to_ram

        self.h5_file = None
        self.driver = driver
        self.h5kwargs = {'driver': driver, **h5kwargs}
        self.stacked = None

    def setup(self):
        shm_dir = f"/dev/shm/{os.getlogin()}"
        if self.driver == 'split':
            self.collection_file = stage_split_hdf5(self.collection_file, stage_dir=shm_dir)
        elif self.to_ram:
            shm_dir = require_directory(shm_dir)
            file_target = os.path.join(shm_dir, os.path.basename(self.collection_file))
            self.collection_file = require_copy(self.collection_file, file_target, follow_symlinks=True)

        comm.barrier()

        log.info(f"Load data from file {self.collection_file}")
        self.h5_file = h5.File(self.collection_file, 'r', **self.h5kwargs)

        self.stacked = self.h5_file.attrs['stack_mods'] if 'stack_mods' in self.h5_file.attrs.keys() else None

    def _center_crop(self, dset: Any, row: int, col: int):
        offs = (self.crop_size[0] // 2, self.crop_size[1] // 2)

        # Fix positions that are not within the section
        r = min(max(row, offs[-2]), dset.shape[-2] - offs[-2])
        c = min(max(col, offs[-1]), dset.shape[-1] - offs[-1])

        santa_crop = dset[..., r - offs[-2]: r + offs[-2], c - offs[-1]: c + offs[-1]]  # hohoho

        return santa_crop

    def __getitem__(self, loc: Location):
        if self.stacked is not None:
            dset = self.h5_file[loc.dataset]
            trans, dir, ret = self._center_crop(dset, loc.row, loc.col).astype(np.float32)
            dir = np.deg2rad(dir)
        else:
            dset = self.h5_file[os.path.join(loc.dataset, 'NTransmittance')]
            trans = np.array(self._center_crop(dset, loc.row, loc.col), dtype=np.float32)
            dset = self.h5_file[os.path.join(loc.dataset, 'Direction')]
            dir = np.deg2rad(self._center_crop(dset, loc.row, loc.col), dtype=np.float32)
            dset = self.h5_file[os.path.join(loc.dataset, 'Retardation')]
            ret = np.array(self._center_crop(dset, loc.row, loc.col), dtype=np.float32)

        return {'trans': trans, 'dir': dir, 'ret': ret}
