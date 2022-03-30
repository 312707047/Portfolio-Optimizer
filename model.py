import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()

class PolicyNetwork(nn.Module):
    def __init__(self) -> None:
        super(PolicyNetwork, self).__init__()


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super(QNetwork).__init__()