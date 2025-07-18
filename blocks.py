import math
import os
import torch
import numpy as np


class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        #return torch.sign(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad):
        return grad.clamp(-1., 1.)


class STLayer(torch.nn.Module):

    def __init__(self):
        super(STLayer, self).__init__()
        self.func = StraightThrough.apply

    def forward(self, x):
        return self.func(x)


class STSigmoid(torch.nn.Module):
    def __init__(self):
        super(STSigmoid, self).__init__()

    def forward(self, x):
        m = torch.distributions.Bernoulli(logits=x)
        sample = m.sample()
        probs = torch.sigmoid(x)
        sample = sample + probs - probs.detach()
        return sample


class Linear(torch.nn.Module):
    """ linear layer with optional batch normalization. """
    def __init__(self, in_features, out_features, std=None, batch_norm=False, gain=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm1d(num_features=self.out_features)

        if std is not None:
            self.weight.data.normal_(0., std)
            self.bias.data.normal_(0., std)
        else:
            # defaults to linear activation
            if gain is None:
                gain = 1
            stdv = math.sqrt(gain / self.weight.size(1))
            self.weight.data.normal_(0., stdv)
            self.bias.data.zero_()

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        if hasattr(self, "batch_norm"):
            x = self.batch_norm(x)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class MLP(torch.nn.Module):
    """ multi-layer perceptron with batch norm option """
    def __init__(self, layer_info, activation=torch.nn.ReLU(), std=None, batch_norm=False, indrop=None, hiddrop=None,
                 last_layer_norm=False):
        super(MLP, self).__init__()
        layers = []
        in_dim = layer_info[0]
        #print(layer_info)
        for i, unit in enumerate(layer_info[1:-1]):
            if i == 0 and indrop:
                layers.append(torch.nn.Dropout(indrop))
            elif i > 0 and hiddrop:
                layers.append(torch.nn.Dropout(hiddrop))
            layers.append(Linear(in_features=in_dim, out_features=unit, std=std, batch_norm=batch_norm, gain=2))
            layers.append(activation)
            in_dim = unit
        if last_layer_norm:
            layers.append(NormedLinear(in_features=in_dim, out_features=layer_info[-1]))
        else:
            layers.append(Linear(in_features=in_dim, out_features=layer_info[-1]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def load(self, path, name):
        state_dict = torch.load(os.path.join(path, name+".ckpt"), weights_only=True)
        self.load_state_dict(state_dict)

    def save(self, path, name):
        dv = self.layers[-1].weight.device
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.cpu().state_dict(), os.path.join(path, name+".ckpt"))
        self.train().to(dv)


class NormedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        stdv = math.sqrt(1 / self.weight.size(1))
        self.weight.data.normal_(0., stdv)

    def forward(self, x):
        x = 3 * torch.nn.functional.normalize(x, dim=-1)
        wn = 3 * torch.nn.functional.normalize(self.weight, dim=-1)
        x = torch.nn.functional.linear(x, wn)
        return x

    def extra_repr(self):
        return "in_features={}, out_features={}".format(self.in_features, self.out_features)


class Flatten(torch.nn.Module):
    def __init__(self, dims):
        super(Flatten, self).__init__()
        self.dims = dims

    def forward(self, x):
        dim = 1
        for d in self.dims:
            dim *= x.shape[d]
        return x.reshape(-1, dim)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        o = x.reshape(self.shape)
        return o


class Avg(torch.nn.Module):
    def __init__(self, dims):
        super(Avg, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims)

    def extra_repr(self):
        return "dims=[" + ", ".join(list(map(str, self.dims))) + "]"


class ChannelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ChannelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)
        h = self.model(x)
        h = h.reshape(B, -1)
        return h
