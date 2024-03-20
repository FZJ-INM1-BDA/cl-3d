from collections import namedtuple
from typing import Tuple, Any, Dict

import os

import torch
import torch.nn.functional as F
import numpy as np
import h5py as h5

from atlaslib.io import stage_split_hdf5
from atlaslib.files import require_copy, require_directory
from plio.section import Section

from cl_3d import utils
from cl_3d.datamodules.components.sampling import Location

# Distributed
#from atlasmpi import MPI


#comm = MPI.COMM_WORLD


log = utils.get_logger(__name__)


Coord = namedtuple("Coord", ('x', 'y'))


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

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

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



class SectionDataset(torch.utils.data.Dataset):

    def __init__(self, trans_file, dir_file, ret_file, patch_shape, out_shape, ram=True, norm_trans=None, norm_ret=None):
        # Expands the dataset to size input by repeating the provided ROIs
        # rois is a list of dicts with entries 'mask', 'ntrans', 'ret' and 'dir'
        super().__init__()
        self.ram = ram
        self.trans_section_mod = Section(path=trans_file)
        self.dir_section_mod = Section(path=dir_file)
        self.ret_section_mod = Section(path=ret_file)
        if ram:
            print("Load sections to RAM...")
            self.trans_section = np.array(self.trans_section_mod.image)
            self.dir_section = np.array(self.dir_section_mod.image)
            self.ret_section = np.array(self.ret_section_mod.image)
            print("All sections loaded to RAM")
        else:
            print("Do not load sections to RAM")
            self.trans_section = self.trans_section_mod.image
            self.dir_section = self.dir_section_mod.image
            self.ret_section = self.ret_section_mod.image

        if norm_trans is None:
            if self.trans_section_mod.norm_value is not None:
                self.norm_trans = self.trans_section_mod.norm_value
            else:
                print("[WARNING] Did not find a normalization value for Transmittance")
                self.norm_trans = 1.0
        else:
            self.norm_trans = norm_trans
            print(f"Normalize Transmittance by value of {self.norm_trans}")
        if norm_ret is None:
            self.norm_ret = 1.0
        else:
            self.norm_ret = norm_ret
            print(f"Normalize Retardation by value of {self.norm_ret}")
        self.brain_id = self.trans_section_mod.brain_id
        self.section_id = self.trans_section_mod.id
        self.section_roi = self.trans_section_mod.roi

        assert (patch_shape[0] - out_shape[0]) % 2 == 0  # Border symmetric
        assert (patch_shape[1] - out_shape[1]) % 2 == 0  # Border symmetric
        self.patch_shape = patch_shape
        self.out_shape = out_shape
        self.border = ((patch_shape[0] - out_shape[0]) // 2, (patch_shape[1] - out_shape[1]) // 2)
        self.shape = self.trans_section.shape

        self.coords = [Coord(x=x, y=y) for x in np.arange(0, self.shape[1], out_shape[1]) for y in
                       np.arange(0, self.shape[0], out_shape[0])]

    def __getitem__(self, i):
        x = self.coords[i].x
        y = self.coords[i].y

        b_y = self.border[0]
        b_x = self.border[1]

        pad_y_0 = max(b_y - y, 0)
        pad_x_0 = max(b_x - x, 0)
        pad_y_1 = max(y + (self.patch_shape[0] - b_y) - self.shape[0], 0)
        pad_x_1 = max(x + (self.patch_shape[1] - b_x) - self.shape[1], 0)

        trans_crop = torch.tensor(np.array(
            self.trans_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        )) / self.norm_trans
        ret_crop = torch.tensor(np.array(
            self.ret_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        )) / self.norm_ret
        dir_crop = torch.tensor(np.deg2rad(
            self.dir_section[max(0, y - b_y):min(self.shape[0], y + self.patch_shape[0] - b_y),
            max(0, x - b_x):min(self.shape[1], x + self.patch_shape[1] - b_x)],
            dtype=np.float32
        ))

        trans_crop = F.pad(trans_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)
        dir_crop = F.pad(dir_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)
        ret_crop = F.pad(ret_crop, (pad_x_0, pad_x_1, pad_y_0, pad_y_1), mode='constant', value=0.0)

        return {'x': x, 'y': y, 'trans': trans_crop[None], 'dir': dir_crop[None], 'ret': ret_crop[None]}

    def __len__(self):
        return len(self.coords)
