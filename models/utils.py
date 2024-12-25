import torch.nn as nn


def create_target_network(source_network: nn.Module) -> nn.Module:
    """
    Creates a target network by copying the parameters from a source network.

    Args:
        source_network (nn.Module): Source network to copy parameters from.

    Returns:
        nn.Module: Target network.
    """
    target_network = type(source_network)(*source_network.args)
    target_network.load_state_dict(source_network.state_dict())
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def update_target_network(target_network: nn.Module, source_network: nn.Module, tau: float = 0.005) -> None:
    """
    Updates the target value network using Polyak averaging.

    Args:
        tau (float): Update rate for target networks.
    """
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
