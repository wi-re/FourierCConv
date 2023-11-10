
# Math/parallelization library includes
import numpy as np
import torch

# Imports for neighborhood searches later on
from torch_geometric.nn import radius


# Neighborhood search
def findNeighborhoods(particles, allParticles, support):
    # Call the external neighborhood search function
    row, col = radius(allParticles, particles, support, max_num_neighbors = 256)
    fluidNeighbors = torch.stack([row, col], dim = 0)
        
    # Compute the distances of all particle pairings
    fluidDistances = (allParticles[fluidNeighbors[1]] - particles[fluidNeighbors[0]])
    # This could also be done with an absolute value function
    fluidRadialDistances = torch.abs(fluidDistances)# torch.sqrt(fluidDistances**2)

    # Compute the direction, in 1D this is either 0 (i == j) or +-1 depending on the relative position
    fluidDistances[fluidRadialDistances < 1e-7] = 0
    fluidDistances[fluidRadialDistances >= 1e-7] /= fluidRadialDistances[fluidRadialDistances >= 1e-7]
    fluidRadialDistances /= support
    
    # Modify the neighbor list so that everything points to the original particles
    particleIndices = torch.arange(particles.shape[0]).to(particles.device)
    stackedIndices = torch.hstack((particleIndices, particleIndices, particleIndices))
    fluidNeighbors[1,:] = stackedIndices[fluidNeighbors[1,:]]    
    
    return fluidNeighbors, fluidRadialDistances, fluidDistances

def periodicNeighborSearch(fluidPositions, particleSupport, minDomain, maxDomain):
    distanceMat = fluidPositions[:,None] - fluidPositions
    distanceMat = torch.remainder(distanceMat + minDomain, maxDomain - minDomain) - maxDomain
    neighs = torch.abs(distanceMat) < particleSupport
    n0 = torch.sum(neighs, dim = 0)
    indices = torch.arange(fluidPositions.shape[0]).to(fluidPositions.device)
    indexMat = indices.expand(fluidPositions.shape[0], fluidPositions.shape[0])
    j, i = indexMat[neighs], indexMat.mT[neighs]
    distances = -distanceMat[neighs]
    directions = torch.sign(distances)    
    return torch.vstack((i, j)), torch.abs(distances)  / particleSupport, directions

    
def batchedNeighborsearch(positions, setup):
    neighborLists = [periodicNeighborSearch(p, s['particleSupport'], s['minDomain'], s['maxDomain']) for p, s in zip(positions, setup)]
    
    neigh_i = [n[0][0] for n in neighborLists]
    neigh_j = [n[0][1] for n in neighborLists]
    neigh_distance = [n[1] for n in neighborLists]
    neigh_direction = [n[2] for n in neighborLists]
    
    for i in range(len(neighborLists) - 1):
        neigh_i[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        neigh_j[i + 1] += np.sum([positions[j].shape[0] for j in range(i+1)])
        
    neigh_i = torch.hstack(neigh_i)
    neigh_j = torch.hstack(neigh_j)
    neigh_distance = torch.hstack(neigh_distance)
    neigh_direction = torch.hstack(neigh_direction)
    
    return neigh_i, neigh_j, neigh_distance, neigh_direction
