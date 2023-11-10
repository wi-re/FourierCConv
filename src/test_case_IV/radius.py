import itertools
import torch
from torch_geometric.nn import radius

from torch.profiler import profile, record_function, ProfilerActivity

def createGhostParticlesKernel3D(positions, domainMin, domainMax, buffer, support, periodic):
    with record_function("createGhostParticlesKernel3D"): 
        indices = torch.arange(positions.shape[0], dtype=torch.int64).to(positions.device)
        virtualMin = domainMin
        virtualMax = domainMax

        masks = []
        offsets = []
        for d in range(3):        
            masks.append(positions[:,d] >= virtualMax[d] - buffer * support)
            masks.append(positions[:,d] < virtualMin[d] + buffer * support)
            shift = virtualMax - virtualMin
            for dd in range(3):
    #             print(d,dd)
                if d != dd:
                    shift[dd] = 0
    #         print(shift)
            offsets.append(-shift)
            offsets.append(shift)
        
        filters = []
        shift = []
        for c in itertools.combinations(zip(offsets, masks), 1):
            filters.append(indices[c[0][1]])
            shift.append(c[0][0])
        for c in itertools.combinations(zip(offsets, masks), 2):
            filters.append(indices[torch.logical_and(c[0][1], c[1][1])])
            shift.append(c[0][0] + c[1][0])
        for c in itertools.combinations(zip(offsets, masks), 3):
            filters.append(indices[torch.logical_and(torch.logical_and(c[0][1], c[1][1]), c[2][1])])
            shift.append(c[0][0] + c[1][0] + c[2][0])
        return filters, shift

def periodicNeighborSearchXYZ(x, ptcls, minDomain, maxDomain, support, periodicX, periodicY, useCompactHashMap = True, searchRadius = 1):
    with record_function("periodicNeighborSearchXYZ"): 
        minD = torch.tensor(minDomain).to(ptcls.device).type(ptcls.dtype)
        maxD = torch.tensor(maxDomain).to(ptcls.device).type(ptcls.dtype)
        y = torch.remainder(ptcls - minD, maxD - minD) + minD
        
        with record_function("Create Ghost Particles Pass"): 
            ghostIndices, ghostOffsets = createGhostParticlesKernel3D(y, torch.tensor([-1,-1,-1]).to(ptcls.device), torch.tensor([1,1,1]).to(ptcls.device), buffer = 2, support = support, periodic = [True, True, True])
        # print(ghostOffsets)
    #     x = gridPositions

    #     y = gridPositions
        indices = torch.cat(ghostIndices)

        positions = torch.cat([y[g] + offset for g, offset in zip(ghostIndices, ghostOffsets)])

        indices = torch.hstack((torch.arange(y.shape[0]).to(y.device), indices))
        y = torch.vstack((y, positions))

        with record_function("Radius Search"): 
            i, j = radius(x,y,support, max_num_neighbors = 256, batch_x = None, batch_y = None)
        with record_function("Neighborsearch Postprocess"): 
            i_t = indices[i]
            
            fluidDistances = y[i] - x[j]
            fluidRadialDistances = torch.linalg.norm(fluidDistances,axis=1)

            fluidDistances[fluidRadialDistances < 1e-4 * support,:] = 0
            fluidDistances[fluidRadialDistances >= 1e-4 * support,:] /= fluidRadialDistances[fluidRadialDistances >= 1e-4 * support,None]
            fluidRadialDistances /= support

        return i_t, j, fluidDistances, fluidRadialDistances#, x, y, ii, ni, jj, nj