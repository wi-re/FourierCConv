import torch
from .util import cpow, getDistancesRel
import numpy as np

@torch.jit.script
def evalRBFSeries(n : int, x : torch.Tensor, which : str = 'linear', epsilon : float = 1., periodic : bool = False, adjustSpacing : bool = False, normalized : bool = False):   
    k = int(epsilon)
    if adjustSpacing:
        if which == 'gaussian' or which == 'inverse_quadric' or which == 'inverse_multiquadric' or 'spline' in which  or 'wendland' in which:
            x = x * (1 - 2/n)
        if which == 'bump':
            x = x * (1 - 4/n)
    
    rRel = getDistancesRel(n, x, periodic)
    r = torch.abs(rRel)
    if n == 1:
        return torch.ones_like(r)
    res = torch.zeros_like(r)
    if not adjustSpacing and not normalized:
        if which == 'linear':               res = torch.clamp(1. - r / epsilon,0,1)
        if which == 'gaussian':             res = torch.exp(-(epsilon * r)**2)
        if which == 'multiquadric':         res = torch.sqrt(1. + (epsilon * r) **2)
        if which == 'inverse_quadric':      res = 1. / ( 1 + (epsilon * r) **2)
        if which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (epsilon * r) **2)
        if which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        if which == 'bump':                 res = torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r))
        if which == 'cubic_spline':         res = cpow(1-r/(epsilon * 1.),3) - 4. * cpow(1/2-r/(epsilon * 1.),3)
        if which == 'quartic_spline':       res = cpow(1-r/(epsilon * 1.),4) - 5 * cpow(3/5-r/(epsilon * 1.),4) + 10 * cpow(1/5-r/(epsilon * 1.),4)
        if which == 'quintic_spline':       res = cpow(1-r/(epsilon * 1.),5) - 6 * cpow(2/3-r/(epsilon * 1.),5) + 15 * cpow(1/3-r/(epsilon * 1.),5)
        if which == 'wendland2':            res = cpow(1 - r/(epsilon * 1.), 4) * (1 + 4 * r/(epsilon * 1.))
        if which == 'wendland4':            res = cpow(1 - r/(epsilon * 1.), 6) * (1 + 6 * r/(epsilon * 1.) + 35/3 * (r/(epsilon * 1.))**2)
        if which == 'wendland6':            res = cpow(1 - r/(epsilon * 1.), 8) * (1 + 8 * r/(epsilon * 1.) + 25 * (r/(epsilon * 1.)) **2 + 32 * (r * (epsilon * 1.))**3)
        if which == 'poly6':                res = cpow(1 - (r/epsilon)**2, 3)
        if which == 'spiky':                res = cpow(1 - r/epsilon, 3)
        if which == 'square':               res = torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    if adjustSpacing and not normalized:
        if which == 'linear':               res = torch.clamp(1. - r / epsilon,0,1)
        if which == 'gaussian':             res = torch.exp(-(epsilon * r)**2)
        if which == 'multiquadric':         res = torch.sqrt(1. + (epsilon * r) **2)
        if which == 'inverse_quadric':      res = 1. / ( 1 + (epsilon * r) **2)
        if which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (epsilon * r) **2)
        if which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        if which == 'bump':                 res = torch.where(r < 1./epsilon, torch.exp(-1./(1- (epsilon * r)**2)), torch.zeros_like(r))
        if which == 'cubic_spline':         res = cpow(1-r/(epsilon * 1.732051),3) - 4. * cpow(1/2-r/(epsilon * 1.732051),3)
        if which == 'quartic_spline':       res = cpow(1-r/(epsilon * 1.936492),4) - 5 * cpow(3/5-r/(epsilon * 1.936492),4) + 10 * cpow(1/5-r/(epsilon * 1.732051),4)
        if which == 'quintic_spline':       res = cpow(1-r/(epsilon * 2.121321),5) - 6 * cpow(2/3-r/(epsilon * 2.121321),5) + 15 * cpow(1/3-r/(epsilon * 2.121321),5)
        if which == 'wendland2':            res = cpow(1 - r/(epsilon * 1.620185), 4) * (1 + 4 * r/(epsilon * 1.620185))
        if which == 'wendland4':            res = cpow(1 - r/(epsilon * 1.936492), 6) * (1 + 6 * r/(epsilon * 1.936492) + 35/3 * (r/(epsilon * 1.936492))**2)
        if which == 'wendland6':            res = cpow(1 - r/(epsilon * 2.207940), 8) * (1 + 8 * r/(epsilon * 2.207940) + 25 * (r/(epsilon * 2.207940)) **2 + 32 * (r * (epsilon * 2.207940))**3)
        if which == 'poly6':                res = cpow(1 - (r/epsilon)**2, 3)
        if which == 'spiky':                res = cpow(1 - r/epsilon, 3)
        if which == 'square':               res = torch.where(torch.logical_and(rRel > -0.5 * epsilon, rRel <= 0.5 * epsilon), torch.ones_like(r), torch.zeros_like(r))
    if not adjustSpacing and normalized:
        if which == 'linear':               res = torch.clamp(1. - r / 1,0,1)
        if which == 'gaussian':             res = torch.exp(-(0.9919394235466537 * r)**2)
        if which == 'multiquadric':         res = torch.sqrt(1. + (1 * r) **2)
        if which == 'inverse_quadric':      res = 1. / ( 1 + (1.1480214948705423 * r) **2)
        if which == 'inverse_multiquadric': res = 1. / torch.sqrt(1. + (1.6382510991695163 * r) **2)
        if which == 'polyharmonic':         res = torch.pow(r, k) if k % 2 == 1 else torch.pow(r,k-1) * torch.log(torch.pow(r,r))
        if which == 'bump':                 res = torch.where(r < 1./0.38739618954567656, torch.exp(-1./(1- (0.38739618954567656 * r)**2)), torch.zeros_like(r))
        if which == 'cubic_spline':         res = cpow(1-r/(epsilon * 2.009770395701026),3) - 4. * cpow(1/2-r/(epsilon * 2.009770395701026),3)
        if which == 'quartic_spline':       res = cpow(1-r/(epsilon * 2.4318514899853443),4) - 5 * cpow(3/5-r/(epsilon * 2.4318514899853443),4) + 10 * cpow(1/5-r/(epsilon * 2.4318514899853443),4)
        if which == 'quintic_spline':       res = cpow(1-r/(epsilon * 2.8903273082559844),5) - 6 * cpow(2/3-r/(epsilon * 2.8903273082559844),5) + 15 * cpow(1/3-r/(epsilon * 2.8903273082559844),5)
        if which == 'wendland2':            res = cpow(1 - r/(epsilon * 3.6238397655105032), 4) * (1 + 4 * r/(epsilon * 3.6238397655105032))
        if which == 'wendland4':            res = cpow(1 - r/(epsilon * 3.7338788470933073), 6) * (1 + 6 * r/(epsilon * 3.7338788470933073) + 35/3 * (r/(epsilon * 3.7338788470933073))**2)
        if which == 'wendland6':            res = cpow(1 - r/(epsilon * 1.3856863702979971), 8) * (1 + 8 * r/(epsilon * 1.3856863702979971) + 25 * (r/(epsilon * 1.3856863702979971)) **2 + 32 * (r * (epsilon * 1.3856863702979971))**3)
        if which == 'poly6':                res = cpow(1 - (r/ 2.6936980947728384)**2, 3)
        if which == 'spiky':                res = cpow(1 - r/3, 3)
        if which == 'square':               res = torch.where(torch.logical_and(rRel > -0.5 * 1, rRel <= 0.5 * 1), torch.ones_like(r), torch.zeros_like(r))
    
    if normalized:
        res = res / torch.sum(res, dim = 0)
    return res
# Evaluate a chebyshev series of the first kind
@torch.jit.script
def evalChebSeries(n : int,x : torch.Tensor):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# Evaluate a chebyshev series of the second kind
@torch.jit.script
def evalChebSeries2(n : int,x : torch.Tensor):
    cs = []
    for i in range(n):
        if i == 0:
            cs.append(torch.ones_like(x))
        elif i == 1:
            cs.append(2 * x)
        else:
            cs.append(2. * x * cs[i-1] - cs[i-2])
    return torch.stack(cs)

# precomputed value for computational efficiency
sqrt_pi_1 = 1. / np.sqrt(np.pi)
sqrt_2pi_1 = 1. / np.sqrt(2. * np.pi)
# Evaluate a fourier series
@torch.jit.script
def fourier(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327

    if n == 0:
        return torch.ones_like(x) * sqrt_2pi_1
    elif n % 2 == 0:
        return torch.cos((n // 2 + 1) * x) * sqrt_pi_1
    return torch.sin((n // 2 + 1) * x) * sqrt_pi_1
@torch.jit.script
def evalFourierSeries(n : int, x : torch.Tensor):
    fs = []
    for i in range(n):
        fs.append(fourier(i, x))
    return torch.stack(fs)

@torch.jit.script
def fourier2(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327
    if n == 0:
        return torch.ones_like(x) * sqrt_2pi_1
    elif n  % 2 == 0:
        return torch.cos(((n - 1) // 2 + 1) * x) * sqrt_pi_1
    return torch.sin(((n-1) // 2 + 1) * x) * sqrt_pi_1
@torch.jit.script
def evalFourierSeries2(n : int, x : torch.Tensor):
    fs = []
    for i in range(n):
        fs.append(fourier2(i, x))
    return torch.stack(fs)

@torch.jit.script
def wrongFourierBasis(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    if n % 2 == 0:
        return (torch.cos((n // 2 + 1) * x) * sqrt_pi_1)
    return (torch.sin((n // 2 + 1) * x) * sqrt_pi_1)
@torch.jit.script
def correctFourierBasis(n : int, x : torch.Tensor):
    sqrt_pi_1 = 0.5641895835477563
    if n % 2 == 0:
        return (torch.cos((n // 2 + 1) * x) * sqrt_pi_1)
    return (torch.sin((n // 2 + 1) * x) * sqrt_pi_1)

@torch.jit.script
def buildFourierSeries(n : int, x : torch.Tensor, kind : str = 'fourier'):
    sqrt_pi_1 = 0.5641895835477563
    sqrt_2pi_1 = 0.3989422804014327
    ndc = True if 'ndc' in kind else False
    fs = []
    for i in range(n):
        if not ndc and i == 0:
            if 'lin' in  kind:
                fs.append(x / 2. * np.pi)
            elif 'sgn' in kind:
                fs.append(torch.sign(x) / 2. * np.pi)
            else:
                fs.append(torch.ones_like(x) * sqrt_2pi_1)
            continue
        if 'odd' in kind:
            fs.append(torch.sin(((i - (0 if ndc else 1)) + 1) * x) * sqrt_pi_1)
        elif 'even' in kind:
            fs.append(torch.cos(((i - (0 if ndc else 1)) + 1) * x) * sqrt_pi_1)
        elif 'ffourier' in kind:
            fs.append(correctFourierBasis(i - (0 if ndc else 1),x))
        else:
            fs.append(wrongFourierBasis(i + (1 if ndc else 0),x))
    return torch.stack(fs)


# Parent function that delegates the call to the corresponding evaluation functions
@torch.jit.script
def evalBasisFunction(n : int, x : torch.Tensor, which : str = 'chebyshev', periodic : bool = False):   
    s = which.split()    
    if s[0] == 'chebyshev':
        return evalChebSeries(n, x)
    if s[0] == 'chebyshev2':
        return evalChebSeries2(n, x)
    if 'fourier' in which:
        return buildFourierSeries(n, x * np.pi, kind = which)
    if s[0] == 'linear':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)        
    if s[0] == 'dmcf':
        return evalRBFSeries(n, x, which = 'linear', epsilon = 1., periodic = periodic)      #torch.sign(x) * evalRBFSeries(n, torch.abs(x) * 2 - 1, which = 'linear', epsilon = 1., periodic = periodic)              
    if s[0] == 'rbf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic)     
    if s[0] == 'abf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, adjustSpacing = True)     
    if s[0] == 'ubf':
        eps = 1. if len(s) < 3 else float(s[2])
        return evalRBFSeries(n, x, which = s[1], epsilon = eps, periodic = periodic, normalized = True)
