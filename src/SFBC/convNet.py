# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# import inspect
# import re
# def debugPrint(x):
#     frame = inspect.currentframe().f_back
#     s = inspect.getframeinfo(frame).code_context[0]
#     r = re.search(r"\((.*)\)", s).group(1)
#     print("{} [{}] = {}".format(r,type(x).__name__, x))
import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from torch_geometric.nn import radius
# from torch.optim import Adam
import copy
import torch
# from torch_geometric.loader import DataLoader
# import argparse
# from torch_geometric.nn import radius
# from torch.optim import Adam
# import matplotlib.pyplot as plt
# import portalocker
# import seaborn as sns
import torch
import torch.nn as nn


# from .detail.cutlass import cutlass
from .convLayer import RbfConv
# from datautils import *
# from plotting import *

# Use dark theme
# from tqdm.autonotebook import trange, tqdm
# import os


class RbfNet(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures = 0, layers = [32,64,64,2], denseLayer = True, activation = 'relu',
                coordinateMapping = 'cartesian', dims = [8], windowFn = None, rbfs = ['linear', 'linear'],batchSize = 32, ignoreCenter = True, normalized = False):
        super().__init__()
        self.centerIgnore = ignoreCenter
        self.features = copy.copy(layers)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')
        self.layers = layers
        self.dims = dims
        self.rbfs = rbfs
        self.dim = len(dims)
        self.normalized = normalized
        self.hasBoundaryLayers = boundaryFeatures != 0
        if len(layers) == 1:
            self.convs.append(RbfConv(
                in_channels = fluidFeatures, out_channels = self.features[0],
                dim = len(dims), size = dims,
                rbf = rbfs,
                bias = False,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))

            self.centerIgnore = False
            return

        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = self.features[0],
            dim = len(dims), size = dims,
            rbf = rbfs,
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
        if boundaryFeatures != 0:
            self.convs.append(RbfConv(
                in_channels = boundaryFeatures, out_channels = self.features[0],
                dim = len(dims), size = dims,
                rbf = rbfs,
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))

        self.fcs.append(nn.Linear(in_features=fluidFeatures,out_features= layers[0],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)

        self.features[0] = self.features[0]
        for i, l in enumerate(layers[1:-1]):
            self.convs.append(RbfConv(
                in_channels = (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]) if i == 0 else self.features[i], out_channels = layers[i+1],
                dim = len(dims), size = dims,
                rbf = rbfs,
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
            self.fcs.append(nn.Linear(in_features=(3 * layers[0] if boundaryFeatures != 0 else 2 * self.features[0]) if i == 0 else layers[i],out_features=layers[i+1],bias=True))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            torch.nn.init.zeros_(self.fcs[-1].bias)
            
        self.convs.append(RbfConv(
            in_channels = self.features[-2] if len(layers) > 2 else (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]), out_channels = self.features[-1],
                dim = len(dims), size = dims,
                rbf = rbfs,
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False, normalizeInterpolation = normalized))
        self.fcs.append(nn.Linear(in_features=self.features[-2] if len(layers) > 2 else (3 * self.features[0] if boundaryFeatures != 0 else 2 * self.features[0]),out_features=self.features[-1],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)


    def forward(self, \
                fluidFeatures, \
                fi, fj, distances, boundaryFeatures = None, bf = None, bb = None, boundaryDistances = None):
        if self.centerIgnore:
            nequals = fi != fj

        i, ni = torch.unique(fi, return_counts = True)
        if self.hasBoundaryLayers:
            b, nb = torch.unique(bf, return_counts = True)
            boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        self.ni = ni
        if self.hasBoundaryLayers:
            self.nb = nb

            ni[i[b]] += nb
        self.li = torch.exp(-1 / 16 * ni)
        if len(self.rbfs) > 2:
            self.li = torch.exp(-1 / 50 * ni)

        if self.centerIgnore:
            fluidEdgeIndex = torch.stack([fi[nequals], fj[nequals]], dim = 0)
        else:
            fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
            
        if self.centerIgnore:
            fluidEdgeLengths = distances[nequals]
        else:
            fluidEdgeLengths = distances
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths))
#         fluidConvolution = scatter_sum(baseArea * fluidFeatures[fluidEdgeIndex[1]] * kernelGradient(torch.abs(fluidEdgeLengths), torch.sign(fluidEdgeLengths), particleSupport), fluidEdgeIndex[0], dim = 0, dim_size = fluidFeatures.shape[0])
        
        if len(self.layers) == 1:
            return fluidConvolution 
        linearOutput = (self.fcs[0](fluidFeatures))
        if self.hasBoundaryLayers:
            boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryDistances))
            ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        else:
            ans = torch.hstack((linearOutput, fluidConvolution))
        
        layers = len(self.convs)
        for i in range(1 if not self.hasBoundaryLayers else 2,layers):
            
            ansc = self.relu(ans)
            
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths)
            ansDense = self.fcs[i - (1 if self.hasBoundaryLayers else 0)](ansc)
            
            
            if self.features[i- (2 if self.hasBoundaryLayers else 1)] == self.features[i-(1 if self.hasBoundaryLayers else 0)] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
        return (ans / 128) if self.dim == 2 else ans
    