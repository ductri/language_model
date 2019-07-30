import torch

from naruto_skills import pytorch_utils


def create_padding_mask(seq_len, max_len):
    """

    :param seq_len: (batch)
    :return: (batch, max_len)
    """
    mask = pytorch_utils.length_to_mask(seq_len, max_len=max_len)
    mask = (mask == 0).float()  # reverse 0 and 1 values
    # (batch, 1, 1, max_len)
    return mask[:, None, None, :]


def create_look_ahead_mask(length):
    """

    :param length:
    :return: (1, 1, length, length)
    """
    mask = torch.ones(length, length)
    mask = mask.triu(diagonal=1)
    # (1, 1, length, length)
    return mask[None, None, :, :]