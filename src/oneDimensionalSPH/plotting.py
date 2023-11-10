# Copyright 2023 <COPYRIGHT HOLDER>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# Math/parallelization library includes
import numpy as np
import torch

# Plotting includes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import NearestNDInterpolator
import scipy
from .sph import createGhostParticles, computeDensity
from ..SFBC.detail.radius import findNeighborhoods

# Plots the given simulation (via simulationStates) and the given timesteps
# This function plots both density and velocity
def plotSimulationState(simulationStates, minDomain, maxDomain, dt, timepoints = []):
    fig, axis = plt.subplots(2, 1, figsize=(9,6), sharex = True, sharey = False, squeeze = False)

    axis[0,0].axvline(minDomain, color = 'black', ls = '--')
    axis[0,0].axvline(maxDomain, color = 'black', ls = '--')
    axis[1,0].axvline(minDomain, color = 'black', ls = '--')
    axis[1,0].axvline(maxDomain, color = 'black', ls = '--')

    axis[1,0].set_xlabel('Position')
    axis[1,0].set_ylabel('Velocity[m/s]')
    axis[0,0].set_ylabel('Density[1/m]')

    def plotTimePoint(i, c, simulationStates, axis, minDomain, maxDomain):
        x = simulationStates[i,0,:]
        xPos = torch.remainder(x + minDomain, maxDomain - minDomain) - maxDomain
        y = simulationStates[i,c,:]
        idx = torch.argsort(xPos)
        axis.plot(xPos[idx].detach().cpu().numpy(), y[idx].detach().cpu().numpy(), label = 't = %1.2g' % (i * dt))
    if timepoints == []:
        plotTimePoint(0,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(0,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4*2,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4*2,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]//4*3,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]//4*3,2, simulationStates, axis[0,0], minDomain, maxDomain)
        
        plotTimePoint(simulationStates.shape[0]-1,1, simulationStates, axis[1,0], minDomain, maxDomain)
        plotTimePoint(simulationStates.shape[0]-1,2, simulationStates, axis[0,0], minDomain, maxDomain)
    else:
        for t in timepoints:
            plotTimePoint(t,1, simulationStates, axis[1,0], minDomain, maxDomain)
            plotTimePoint(t,2, simulationStates, axis[0,0], minDomain, maxDomain)

    axis[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()

# # Plots the density of a set of particles, as determined using SPH, as well as the FFT and PSD of the density field
# # Used for plotting the initial density field/sampling
def plotDensityField(fluidPositions, fluidAreas, minDomain, maxDomain, particleSupport):
    ghostPositions = createGhostParticles(fluidPositions, minDomain, maxDomain)
    fluidNeighbors, fluidRadialDistances, fluidDistances = findNeighborhoods(fluidPositions, ghostPositions, particleSupport)
    fluidDensity = computeDensity(fluidPositions, fluidAreas, particleSupport, fluidRadialDistances, fluidNeighbors)

    xs = fluidPositions.detach().cpu().numpy()
    densityField = fluidDensity.detach().cpu().numpy()
    fig, axis = plt.subplots(1, 3, figsize=(18,6), sharex = False, sharey = False, squeeze = False)
    numSamples = densityField.shape[-1]
    fs = numSamples/2
    fftfreq = np.fft.fftshift(np.fft.fftfreq(xs.shape[-1], 1/fs/1))    
    x = densityField
    y = np.abs(np.fft.fftshift(np.fft.fft(x) / len(x)))
    axis[0,0].plot(xs, densityField)
    axis[0,1].loglog(fftfreq[fftfreq.shape[0]//2:],y[fftfreq.shape[0]//2:], label = 'baseTarget')
    f, Pxx_den = scipy.signal.welch(densityField, fs, nperseg=len(x)//32)
    axis[0,2].loglog(f, Pxx_den, label = 'baseTarget')
    axis[0,2].set_xlabel('frequency [Hz]')
    axis[0,2].set_ylabel('PSD [V**2/Hz]')
    fig.tight_layout()
    return fluidDensity
# Plots a pseudo 2D plot of the given simulation showing both the density and velocity (color mapped) with
# time on the y-axis and position on the x-axis to demonstrate how the simulations evolve over time.
# This function is relatively slow as it does a resampling from the particle data to a regular grid, of size
# nx * ny, using NearestNDInterpolator.
def regularPlot(simulationStates, minDomain, maxDomain, dt, nx = 512, ny = 2048):
    timeArray = torch.arange(simulationStates.shape[0])[:,None].repeat(1,simulationStates.shape[2]) * dt
    positionArray = simulationStates[:,0]
    positionArray = torch.remainder(positionArray + minDomain, maxDomain - minDomain) - maxDomain
    
    xys = torch.vstack((timeArray.flatten().to(positionArray.device).type(positionArray.dtype), positionArray.flatten())).mT.detach().cpu().numpy()

    interpVelocity = NearestNDInterpolator(xys, simulationStates[:,1].flatten().detach().cpu().numpy())
    interpDensity = NearestNDInterpolator(xys, simulationStates[:,2].flatten().detach().cpu().numpy())

    X = torch.linspace(torch.min(timeArray), torch.max(timeArray), ny).detach().cpu().numpy()
    Y = torch.linspace(minDomain, maxDomain, nx).detach().cpu().numpy()
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    # Z = interp(X, Y)

    fig, axis = plt.subplots(2, 1, figsize=(14,9), sharex = False, sharey = False, squeeze = False)


    im = axis[0,0].pcolormesh(X,Y,interpDensity(X,Y), cmap = 'viridis', vmin = torch.min(torch.abs(simulationStates[:,2])),vmax = torch.max(torch.abs(simulationStates[:,2])))
    # im = axis[0,0].imshow(simulationStates[:,2].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain])
    axis[0,0].set_aspect('auto')
    axis[0,0].set_xlabel('time[/s]')
    axis[0,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[0,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[0,0].axhline(minDomain, color = 'black', ls = '--')
    axis[0,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Density [1/m]')

    im = axis[1,0].pcolormesh(X,Y,interpVelocity(X,Y), cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    # im = axis[1,0].imshow(simulationStates[:,1].mT, extent = [0,dt * simulationStates.shape[0], maxDomain,minDomain], cmap = 'RdBu', vmin = -torch.max(torch.abs(simulationStates[:,1])),vmax = torch.max(torch.abs(simulationStates[:,1])))
    axis[1,0].set_aspect('auto')
    axis[1,0].set_xlabel('time[/s]')
    axis[1,0].set_ylabel('position')
    ax1_divider = make_axes_locatable(axis[1,0])
    cax1 = ax1_divider.append_axes("right", size="2%", pad="2%")
    cb1 = fig.colorbar(im, cax=cax1,orientation='vertical')
    cb1.ax.tick_params(labelsize=8) 
    axis[1,0].axhline(minDomain, color = 'black', ls = '--')
    axis[1,0].axhline(maxDomain, color = 'black', ls = '--')
    cb1.ax.set_xlabel('Velocity [m/s]')

    fig.tight_layout()
