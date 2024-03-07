from typing import Callable

import torch
import torch.nn.functional as F


def cos_sim(
        x1: torch.Tensor,
        x2: torch.Tensor
) -> torch.Tensor:
    """
    Calculates similarity matrix based on the cosine similarity measure

    Parameters
    ----------
    :param x1: torch.Tensor
        First input.
    :param x2: torch.Tensor
        Second input.

    Returns
    -------
    :return:
        S: torch.Tensor
            Similarity matrix based on the cosine similarity measure
    """

    return F.cosine_similarity(x1.view(-1, 1, x1.shape[-1]), x2.view(1, -1, x2.shape[-1]), dim=-1)


def contrastive_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        tau: float = 1.0,
        sim: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cos_sim
):
    """

    Contrastive loss of the SimCLR paper "A Simple Framework for Contrastive
    Learning of Visual Representations".

    The code is inspired by
    https://github.com/google-research/simclr/blob/master/objective.py

    Parameters
    ----------
    :param z1: torch.Tensor
        Anchor representations where contrastive loss is applied on.
    :param z2: torch.Tensor
        Positive representations where contrastive loss is applied on.
    :param tau: float
        Temperature parameter.
    :param sim: function
        Callable that takes two tensors and calculates the similarity matrix between two torch.Tensor inputs.

    Returns
    -------
    :return:
        loss: torch.Tensor
            The contrastive loss.
    """

    mask = F.one_hot(torch.arange(len(z1), device=z1.device), len(z1))

    # Similarities within originals and transforms
    logits11 = sim(z1, z1) / tau
    logits22 = sim(z2, z2) / tau

    # Eliminate tuples where i=j
    logits11 = logits11 - mask * 1e9
    logits22 = logits22 - mask * 1e9

    # Cross similarity between originals and transforms
    logits12 = sim(z1, z2) / tau
    logits21 = sim(z2, z1) / tau

    # Softmax cross entropy loss
    labels = torch.arange(len(z1), device=z1.device)
    loss1 = F.cross_entropy(torch.cat([logits12, logits11], dim=1), labels)
    loss2 = F.cross_entropy(torch.cat([logits21, logits22], dim=1), labels)

    # Final loss
    loss = (loss1 + loss2) / 2

    return loss