# import os
# import sys
# module_path = os.path.abspath(os.path.join('../../'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
    
# sph related imports
# from src.oneDimensionalSPH.sph import *
# from src.oneDimensionalSPH.perlin import *
# neural network rlated imports
from torch.optim import Adam
# from src.oneDimensionalSPH.rbfConv import *
from torch_geometric.loader import DataLoader
# from src.oneDimensionalSPH.trainingHelper import *
# plotting/UI related imports
from src.oneDimensionalSPH.plotting import *
import matplotlib as mpl
# plt.style.use('dark_background')
cmap = mpl.colormaps['viridis']
from tqdm.autonotebook import trange, tqdm
from IPython.display import display, Latex
from datetime import datetime
# from src.oneDimensionalSPH.rbfNet import *
from src.SFBC.convNet import RbfNet
import h5py
import matplotlib.colors as colors
import torch
import torch.nn as nn
# %matplotlib notebook

from src.oneDimensionalSPH.io import loadFile

def plotTrainingFiles(trainingFiles, numParticles, dt, timesteps):
    ns = int(np.sqrt(len(trainingFiles)))
    fig, axis = plt.subplots(ns, ns, figsize=(ns*6,ns * 2), sharex = True, sharey = True, squeeze = False)

    def plot(fig, axis, mat, title, cmap = 'viridis'):
        im = axis.imshow(mat, extent = [0,dt * timesteps,numParticles,0], cmap = cmap)
        axis.axis('auto')
        ax1_divider = make_axes_locatable(axis)
        cax1 = ax1_divider.append_axes("right", size="2%", pad="6%")
        cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
        cb1.ax.tick_params(labelsize=8) 
        axis.set_title(title)

    for i in range(ns):
        for j in range(ns):
            data = loadFile(trainingFiles[ns * j + i], False)
            plot(fig,axis[i,j], data['velocity'].mT, trainingFiles[ns * j + i].split('/')[-1].split('.')[0].split('_')[2], 'RdBu')
    #         plot(fig,axis[i,j], data['dudt'].mT, trainingFiles[ns * j + i].split('/')[-1].split('.')[0].split('_')[2], 'RdBu')

    fig.tight_layout()



# def plotBatch(trainingData, settings, dataSet, bdata, device, offset, particleSupport, groundTruthFn, featureFn, model = None):
#     positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
#     i, j, distance, direction = batchedNeighborsearch(positions, setup)
#     x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

#     x = x[:,None].to(device)    
#     groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction, particleSupport).to(device)
#     distance = (distance * direction)[:,None].to(device)
#     features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)

# #     optimizer.zero_grad()
# #     prediction = model(features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
# #     lossTerm = lossFn(prediction, groundTruth)
# #     loss = torch.mean(lossTerm)
    
#     fig, axis = plt.subplot_mosaic('''AF
#     BC
#     DE''', figsize=(12,8), sharey = False, sharex = False)
    
#     positions = torch.vstack(positions).mT.detach().cpu().numpy()
#     vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
#     axis['A'].set_title('Position')
#     axis['A'].plot(positions)
#     axis['B'].set_title('Density')
#     axis['B'].plot(positions, density)
#     axis['C'].set_title('Difference')
#     axis['C'].plot(positions, gt - ft)
#     axis['D'].set_title('Instantenous Velocity')
#     axis['D'].plot(positions, vel)
#     axis['E'].set_title('Ground Truth')
#     axis['E'].plot(positions, gt)
#     axis['F'].set_title('Features[:,0]')
#     axis['F'].plot(positions, ft)
    
#     fig.tight_layout()
    
# def plotTrainedBatch(trainingData, settings, dataSet, bdata, device, offset, modelState, groundTruthFn, featureFn, lossFn):
#     positions, velocities, areas, dudts, density, inputVelocity, outputVelocity, setup = loadBatch(trainingData, settings, dataSet, bdata, device, offset)
#     i, j, distance, direction = batchedNeighborsearch(positions, setup)
#     x, u, v, rho, dudt, inVel, outVel = flatten(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity)

#     x = x[:,None].to(device)    
#     groundTruth = groundTruthFn(positions, velocities, areas, density, dudts, inputVelocity, outputVelocity, i, j, distance, direction).to(device)
#     distance = (distance * direction)[:,None].to(device)
#     features = featureFn(x, u, v, rho, dudt, inVel, outVel).to(device)
    
#     with torch.no_grad():
#         prediction = modelState['model'](features.to(device), i.to(device), j.to(device), distance.to(device))[:,0]
#         lossTerm = lossFn(prediction, groundTruth)
#         loss = torch.mean(lossTerm)
    
#     fig, axis = plt.subplot_mosaic('''ABC
#     DEF''', figsize=(16,5), sharey = False, sharex = True)
    
#     positions = torch.vstack(positions).mT.detach().cpu().numpy()
#     vel = u.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     area = v.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     dudt = dudt.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     density = rho.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     inVel = inVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     outVel = outVel.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     gt = groundTruth.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     ft = features[:,0].reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     loss = lossTerm.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
#     pred = prediction.reshape(positions.transpose().shape).mT.detach().cpu().numpy()
    
#     axis['A'].set_title('Density')
#     axis['A'].plot(positions, density)
#     axis['E'].set_title('Ground Truth - Features[:,0]')
#     axis['E'].plot(positions, gt - ft)
#     axis['B'].set_title('Ground Truth')
#     axis['B'].plot(positions, gt)
#     axis['D'].set_title('Features[:,0]')
#     axis['D'].plot(positions, ft)
#     axis['C'].set_title('Prediction')
#     axis['C'].plot(positions, pred)
#     axis['F'].set_title('Loss')
#     axis['F'].plot(positions, loss)
    
#     fig.tight_layout()
    
# # plotBatch(trainingData, settings, dataSet, bdata, device, offset)