import os
import torch
from torch import nn

def get_model_size_weights(mdl):
    """
    Calculates the size of the model's weights in megabytes.

    Args:
        mdl (torch.nn.Module): The model whose weights size needs to be calculated.

    Returns:
        float: The size of the model's weights in megabytes.
    """
    torch.save(mdl.state_dict(), "tmp.pt")
    mdl_size = round(os.path.getsize("tmp.pt") / 1e6, 3)
    os.remove("tmp.pt")
    return mdl_size

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    Calculates the total number of parameters in a given PyTorch model.

    :param model (nn.Module): The PyTorch model.
    :param count_nonzero_only (bool, optional): If True, only counts the number of non-zero parameters.
                                                If False, counts all parameters. Defaults to False.

    """
    
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements
