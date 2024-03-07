from typing import Any, List, Dict

import torch
from pytorch_lightning import LightningModule
import hydra

from dms import DMS

from cl_3d.models.components.layers import GatherLayer
from cl_3d.utils.loss import contrastive_loss
from cl_3d import utils

log = utils.get_logger(__name__)


class ContrastiveLitModule(LightningModule):

    def __init__(
            self,
            encoder: Dict,
            projection: Dict,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            tau: float = 0.5,
            sync_loss: bool = False,
            dms_timeout: int = 0,
            **kwargs
    ):
        super().__init__()

        # Dont save encoder and projection parameters (blows up log file)
        self.save_hyperparameters(logger=False)

        self.encoder = hydra.utils.instantiate(encoder)
        self.projection = hydra.utils.instantiate(projection)

        self.dms = None

    def on_train_start(self):
        if self.hparams.dms_timeout > 0:
            self.dms = DMS(timeout=self.hparams.dms_timeout)

    def on_train_end(self):
        if self.dms:
            self.dms.stop()

    def forward(
            self,
            trans: torch.Tensor,
            dir: torch.Tensor,
            ret: torch.Tensor
    ):
        h = self.encoder(trans, dir, ret)[-1]
        z = self.projection(h)

        return z, h

    def step(self, batch: Any):
        anch, pos, anch_loc, pos_loc = batch
        N, _, H, W = anch['trans'].shape
        stack = dict((m, torch.cat((anch[m], pos[m]), dim=0)) for m in anch.keys())

        z_out, h_out = self.forward(**stack)

        z1, z2 = torch.split(z_out, N)
        if self.hparams.sync_loss:
            z1 = torch.cat(GatherLayer.apply(z1), dim=0)
            z2 = torch.cat(GatherLayer.apply(z2), dim=0)

        loss = contrastive_loss(z1, z2, tau=self.hparams.tau)

        return loss, z1, z2

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _ = self.step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=False, rank_zero_only=True)

        if self.dms:
            self.dms.keep_alive()

        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, z1, z2 = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, rank_zero_only=True)

        # TODO: Return z1, h1

        return {'loss': loss}

    def validation_epoch_end(self, outputs: List[Any]):
        # Gather h1 and z1 and vis as tb pca

        # TODO Represent tensorboard graph

        pass

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optim
