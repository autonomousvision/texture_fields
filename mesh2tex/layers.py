import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ResnetBlockFC(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Resnet Blocks
class ResnetBlockConv1D(nn.Module):
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockPointwise(nn.Module):
    def __init__(self, f_in, f_out=None, f_hidden=None,
                 is_bias=True, actvn=F.relu, factor=1., eq_lr=False):
        super().__init__()
        # Filter dimensions
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out

        self.factor = factor
        self.eq_lr = eq_lr

        # Activation function
        self.actvn = actvn

        # Submodules
        self.conv_0 = nn.Conv1d(f_in, f_hidden, 1)
        self.conv_1 = nn.Conv1d(f_hidden, f_out, 1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv1d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        # Initialization
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        net = self.conv_0(self.actvn(x))
        dx = self.conv_1(self.actvn(net))
        x_s = self.shortcut(x)
        return x_s + self.factor * dx


class ResnetBlockConv2d(nn.Module):
    def __init__(self, f_in, f_out=None, f_hidden=None,
                 is_bias=True, actvn=F.relu, factor=1.,
                 eq_lr=False, pixel_norm=False):
        super().__init__()
        # Filter dimensions
        if f_out is None:
            f_out = f_in

        if f_hidden is None:
            f_hidden = min(f_in, f_out)

        self.f_in = f_in
        self.f_hidden = f_hidden
        self.f_out = f_out
        self.factor = factor
        self.eq_lr = eq_lr
        self.use_pixel_norm = pixel_norm

        # Activation
        self.actvn = actvn

        # Submodules
        self.conv_0 = nn.Conv2d(self.f_in, self.f_hidden, 3,
                                stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.f_hidden, self.f_out, 3,
                                stride=1, padding=1, bias=is_bias)

        if self.eq_lr:
            self.conv_0 = EqualizedLR(self.conv_0)
            self.conv_1 = EqualizedLR(self.conv_1)

        if f_in == f_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Conv2d(f_in, f_out, 1, bias=False)
            if self.eq_lr:
                self.shortcut = EqualizedLR(self.shortcut)

        # Initialization
        nn.init.zeros_(self.conv_1.weight)

    def forward(self, x):
        x_s = self.shortcut(x)

        if self.use_pixel_norm:
            x = pixel_norm(x)
        dx = self.conv_0(self.actvn(x))

        if self.use_pixel_norm:
            dx = pixel_norm(dx)
        dx = self.conv_1(self.actvn(dx))

        out = x_s + self.factor * dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


class EqualizedLR(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self._make_params()

    def _make_params(self):
        weight = self.module.weight

        height = weight.data.shape[0]
        width = weight.view(height, -1).data.shape[1]

        # Delete parameters in child
        del self.module._parameters['weight']
        self.module.weight = None

        # Add parameters to myself
        self.weight = nn.Parameter(weight.data)

        # Inherit parameters
        self.factor = np.sqrt(2 / width)

        # Initialize
        nn.init.normal_(self.weight)

        # Inherit bias if available
        self.bias = self.module.bias
        self.module.bias = None

        if self.bias is not None:
            del self.module._parameters['bias']
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        self.module.weight = self.factor * self.weight
        if self.bias is not None:
            self.module.bias = 1. * self.bias
        out = self.module.forward(*args, **kwargs)
        self.module.weight = None
        self.module.bias = None
        return out


def pixel_norm(x):
    sigma = x.norm(dim=1, keepdim=True)
    out = x / (sigma + 1e-5)
    return out
