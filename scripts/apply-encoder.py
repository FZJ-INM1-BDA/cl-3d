import os
import re
from typing import List, Tuple
from glob import glob
import math

import numpy as np
import h5py as h5
from tqdm import tqdm
import click

from torch.utils.data import DataLoader
import torch

from cl_3d.datamodules.components.data import SectionDataset
from cl_3d.models.contrastive_module import ContrastiveLitModule

# Distributed
from atlasmpi import MPI

comm = MPI.COMM_WORLD


def get_files(
        trans: str,
        dir: str,
        ret: str,
        out: str,
        rank: int = 0,
        size: int = 1
):
    trans_files = sorted(glob(trans))
    dir_files = sorted(glob(dir))
    ret_files = sorted(glob(ret))

    if os.path.isdir(out):
        ft_files = []
        for d_f in dir_files:
            d_fname = os.path.splitext(os.path.basename(d_f))[0]
            d_base = os.path.splitext(d_fname)[0]
            ft_file = re.sub("direction", "Features", d_base, flags=re.IGNORECASE)
            if "Features" not in ft_file:
                ft_file += "_Features.h5"
            else:
                ft_file += ".h5"
            ft_files.append(os.path.join(out, ft_file))
    else:
        ft_files = [out]

    for i, (trans_file, dir_file, ret_file, ft_file) \
            in enumerate(zip(trans_files, dir_files, ret_files, ft_files)):
        if i % size == rank:
            if not os.path.isfile(ft_file):
                yield trans_file, dir_file, ret_file, ft_file
            else:
                print(f"{ft_file} already exists. Skip.")


def create_features(
        encoder: torch.nn.Module,
        section_loader: DataLoader,
        h_size: int,
        z_size: int,
        out_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        rank: int
):
    print("Initialize output featuremaps...")
    h_features = np.zeros((*out_size, h_size), dtype=np.float32)
    z_features = np.zeros((*out_size, z_size), dtype=np.float32)

    def get_outputs(batch, network):
        with torch.no_grad():
            network.eval()
            z, h = network(
                trans=batch['trans'].to(network.device),
                dir=batch['dir'].to(network.device),
                ret=batch['ret'].to(network.device)
            )
        return {'x': batch['x'], 'y': batch['y'], 'h': h, 'z': z}

    def transfer(batch, network):
        b = get_outputs(batch, network)
        for x, y, h, z in zip(b['x'], b['y'], b['h'], b['z']):
            try:
                h_features[y // stride[0], x // stride[1]] = h.cpu().numpy()
            except:
                raise Exception(f"ERROR creating mask at x={x}, y={y}, shape={h_features.shape}")
            try:
                z_features[y // stride[0], x // stride[1]] = z.cpu().numpy()
            except:
                raise Exception(f"ERROR creating mask at x={x}, y={y}, shape={z_features.shape}")

    print("Start feature generation...")
    for batch in tqdm(section_loader, desc=f"Rank {rank}"):
        transfer(batch, encoder)

    return h_features, z_features


def save_features(
        h_features: np.ndarray,
        z_features: np.ndarray,
        ft_file: str,
        spacing: Tuple[float, ...] = (1.0, 1.0),
        origin: Tuple[float, ...] = (0.0, 0.0),
        dtype: str = None,
):
    print("Save features...")
    with h5.File(ft_file, "w") as f:
        feature_group = f.create_group("Features")
        dset_h = feature_group.create_dataset(f"{h_features.shape[-1]}", data=h_features.transpose(2, 0, 1), dtype=dtype)
        dset_h.attrs['spacing'] = spacing
        dset_h.attrs['origin'] = origin
        dset_z = feature_group.create_dataset(f"{z_features.shape[-1]}", data=z_features.transpose(2, 0, 1), dtype=dtype)
        dset_z.attrs['spacing'] = spacing
        dset_z.attrs['origin'] = origin
    print(f"Featuremaps created at {ft_file}")


@click.command()
@click.option("--ckpt", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--trans", type=str)
@click.option("--dir", type=str)
@click.option("--ret", type=str)
@click.option("--out", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--norm_trans", type=float, default=None)
@click.option("--num_workers", type=int, default=0)
@click.option("--batch_size", type=int, default=16)
@click.option("--patch_size", type=int, default=128)
@click.option("--overlap", type=float, default=0.5)
@click.option("--ram", default=False, is_flag=True)
@click.option("--dtype", type=str, default=None)
def cli(ckpt, trans, dir, ret, out, norm_trans, num_workers, batch_size, patch_size, overlap, ram, dtype):
    rank = comm.Get_rank()
    size = comm.size

    if torch.cuda.is_available():
        available_gpus = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        print(f"Found {len(available_gpus)} GPUs")
        device = available_gpus[rank % len(available_gpus)]
    else:
        device = 'cpu'
    print(f"Use device {device} on rank {rank}")

    # Create model
    encoder = ContrastiveLitModule.load_from_checkpoint(ckpt)
    encoder.to(device)
    print(f"Model loaded on rank {rank}")

    patch_shape = (patch_size, patch_size)
    stride = (int((1 - overlap) * patch_size), int((1 - overlap) * patch_size))

    h_size = encoder.projection.features[0]
    z_size = encoder.projection.features[-1]

    for trans_file, dir_file, ret_file, ft_file in get_files(trans, dir, ret, out, rank, size):
        print(f"Initialize DataLoader for {trans_file}, {dir_file}, {ret_file}")

        section_dataset = SectionDataset(trans_file=trans_file, dir_file=dir_file, ret_file=ret_file,
                                         patch_shape=patch_shape, out_shape=stride, ram=ram,
                                         norm_trans=norm_trans)
        section_loader = DataLoader(section_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers)

        out_size = tuple(math.ceil(s / stride[i]) for i, s in enumerate(section_dataset.shape))

        h_features, z_features = create_features(encoder, section_loader, h_size, z_size, out_size, stride, rank)

        spacing = tuple(stride[i] * s for i, s in enumerate(section_dataset.trans_section_mod.spacing))
        origin = section_dataset.trans_section_mod.origin

        save_features(h_features, z_features, ft_file, spacing, origin, dtype)


if __name__ == '__main__':
    cli()
