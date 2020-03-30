import torch.nn as nn
from torchvision import models


class Resnet18(nn.Module):
    ''' ResNet-18 conditioning network.
    '''
    def __init__(self, c_dim=128, normalize=True, use_linear=True):
        ''' Initialisation.

        Args:
            c_dim (int): output dimension of the latent embedding
            normalize (bool): whether the input images should be normalized
            use_linear (bool): whether a final linear layer should be used
        '''
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


def normalize_imagenet(x):
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x
