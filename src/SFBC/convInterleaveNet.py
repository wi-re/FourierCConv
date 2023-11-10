# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import inspect
import re
def debugPrint(x):
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} [{}] = {}".format(r,type(x).__name__, x))
import torch
from torch_geometric.loader import DataLoader
import argparse
from torch_geometric.nn import radius
from torch.optim import Adam
import copy
import torch
from torch_geometric.loader import DataLoader
import argparse
from torch_geometric.nn import radius
from torch.optim import Adam
import matplotlib.pyplot as plt
import portalocker
import seaborn as sns
import torch
import torch.nn as nn


from .detail.cutlass import *
from .convLayer import *
# from datautils import *
# from plotting import *

# Use dark theme
from tqdm.autonotebook import trange, tqdm
import os
           
class RbfInterleaveNet(torch.nn.Module):
    def __init__(self, fluidFeatures, boundaryFeatures, layers = [32,64,64,2], denseLayer = True, activation = 'relu',
                coordinateMapping = 'polar', n = 8, m = 8, windowFn = None, rbf_x = 'linear', rbf_y = 'linear', batchSize = 32, ignoreCenter = True):
        super().__init__()
        self.centerIgnore = ignoreCenter
        self.features = copy.copy(layers)
        self.convs = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.relu = getattr(nn.functional, 'relu')

        self.convs.append(RbfConv(
            in_channels = fluidFeatures, out_channels = self.features[0],
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_x],
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.convs.append(RbfConv(
            in_channels = boundaryFeatures, out_channels = self.features[0],
            dim = 2, size = [n,m],
            rbf = [rbf_x, rbf_x],
            bias = True,
            linearLayer = False, biasOffset = False, feedThrough = False,
            preActivation = None, postActivation = None,
            coordinateMapping = coordinateMapping,
            batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        
        self.fcs.append(nn.Linear(in_features=fluidFeatures,out_features= layers[0],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)

        self.features[0] = self.features[0]
        for i, l in enumerate(layers[1:-1]):
            self.convs.append(RbfConv(
                in_channels = (3 * self.features[0]) if i == 0 else self.features[i], out_channels = layers[i+1],
                dim = 2, size = [n,m],
                rbf = [rbf_y, rbf_y] if i % 2 == 0 else [rbf_x, rbf_x],
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
            self.fcs.append(nn.Linear(in_features=3 * layers[0] if i == 0 else layers[i],out_features=layers[i+1],bias=True))
            torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
            torch.nn.init.zeros_(self.fcs[-1].bias)
            
        self.convs.append(RbfConv(
            in_channels = self.features[-2], out_channels = self.features[-1],
                dim = 2, size = [n,m],
                rbf = [rbf_y, rbf_y] if len(layers)%2 == 1 else [rbf_x, rbf_x],
                bias = True,
                linearLayer = False, biasOffset = False, feedThrough = False,
                preActivation = None, postActivation = None,
                coordinateMapping = coordinateMapping,
                batch_size = [batchSize, batchSize], windowFn = windowFn, normalizeWeights = False))
        self.fcs.append(nn.Linear(in_features=layers[-2],out_features=self.features[-1],bias=True))
        torch.nn.init.xavier_uniform_(self.fcs[-1].weight)
        torch.nn.init.zeros_(self.fcs[-1].bias)


    def forward(self, \
                fluidPositions, boundaryPositions, \
                fluidFeatures, boundaryFeatures,\
                attributes, fluidBatches = None, boundaryBatches = None):
        fi, fj = radius(fluidPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = fluidBatches, batch_y = fluidBatches)
        bf, bb = radius(boundaryPositions, fluidPositions, attributes['support'], max_num_neighbors = 256, batch_x = boundaryBatches, batch_y = fluidBatches)
        if self.centerIgnore:
            nequals = fi != fj

        i, ni = torch.unique(fi, return_counts = True)
        b, nb = torch.unique(bf, return_counts = True)
        
        self.ni = ni
        self.nb = nb

        ni[i[b]] += nb
        self.li = torch.exp(-1 / 16 * ni)
        
        boundaryEdgeIndex = torch.stack([bf, bb], dim = 0)
        boundaryEdgeLengths = (boundaryPositions[boundaryEdgeIndex[1]] - fluidPositions[boundaryEdgeIndex[0]])/attributes['support']
        boundaryEdgeLengths = boundaryEdgeLengths.clamp(-1,1)
        if self.centerIgnore:
            fluidEdgeIndex = torch.stack([fi[nequals], fj[nequals]], dim = 0)
        else:
            fluidEdgeIndex = torch.stack([fi, fj], dim = 0)
        fluidEdgeLengths = -(fluidPositions[fluidEdgeIndex[1]] - fluidPositions[fluidEdgeIndex[0]])/attributes['support']
        fluidEdgeLengths = fluidEdgeLengths.clamp(-1,1)
            
        linearOutput = (self.fcs[0](fluidFeatures))
        boundaryConvolution = (self.convs[1]((fluidFeatures, boundaryFeatures), boundaryEdgeIndex, boundaryEdgeLengths))
        fluidConvolution = (self.convs[0]((fluidFeatures, fluidFeatures), fluidEdgeIndex, fluidEdgeLengths))
        ans = torch.hstack((linearOutput, fluidConvolution, boundaryConvolution))
        if verbose:
            print('first layer output', ans[:4])
        
        layers = len(self.convs)
        for i in range(2,layers):
            
            ansc = self.relu(ans)
            
            ansConv = self.convs[i]((ansc, ansc), fluidEdgeIndex, fluidEdgeLengths)
            ansDense = self.fcs[i - 1](ansc)
            
            
            if self.features[i-2] == self.features[i-1] and ans.shape == ansConv.shape:
                ans = ansConv + ansDense + ans
            else:
                ans = ansConv + ansDense
            if verbose:
                print('\tlayer output after activation', ans[:4])
        return ans / 128