#!/usr/bin/python3

#################################################################################
# Fedorenko
# 
# Copyright (C) 2021, Roshan J. Samuel
#
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################

# Import all necessary modules
import cupy as cp
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

############################### GLOBAL VARIABLES ################################

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = np.array([7, 7, 7])

# Depth of each V-cycle in multigrid
VDepth = min(sInd) - 1

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 3

# Number of iterations during post-smoothing
pstSm = 3

# Tolerance value for iterative solver
tolerance = 1.0e-6

# N should be of the form 2^n
# Then there will be 2^n + 2 points in total, including 2 ghost points
sLst = [2**x for x in range(12)]

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

# Define array of grid spacings along X
h0 = 1.0/(N[0][0])
hx = [h0*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Y
h0 = 1.0/(N[0][1])
hy = [h0*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Z
h0 = 1.0/(N[0][2])
hz = [h0*(2**x) for x in range(VDepth+1)]

# Square of hx, used in finite difference formulae
hx2 = [x*x for x in hx]

# Square of hy, used in finite difference formulae
hy2 = [x*x for x in hy]

# Square of hz, used in finite difference formulae
hz2 = [x*x for x in hz]

# Cross product of hy and hz, used in finite difference formulae
hyhz = [hy2[i]*hz2[i] for i in range(VDepth + 1)]

# Cross product of hx and hz, used in finite difference formulae
hzhx = [hx2[i]*hz2[i] for i in range(VDepth + 1)]

# Cross product of hx and hy, used in finite difference formulae
hxhy = [hx2[i]*hy2[i] for i in range(VDepth + 1)]

# Cross product of hx, hy and hz used in finite difference formulae
hxhyhz = [hx2[i]*hy2[i]*hz2[i] for i in range(VDepth + 1)]

# Factor in denominator of Gauss-Seidel iterations
gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i])) for i in range(VDepth + 1)]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*N[-1][0]*N[-1][1]*N[-1][2]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0

# Flag to determine if non-zero homogenous BC has to be applied or not
zeroBC = False

##################################### MAIN ######################################

def main():
    global N
    global pData
    global rData, sData, iTemp

    nList = np.array(N)

    pData = [cp.zeros(tuple(x)) for x in nList + 2]

    rData = [cp.zeros_like(x) for x in pData]
    sData = [cp.zeros_like(x) for x in pData]
    iTemp = [cp.zeros_like(x) for x in pData]

    initDirichlet()

    mgRHS = cp.ones_like(pData[0])

    # Solve
    t1 = datetime.now()
    mgLHS = multigrid(mgRHS)
    cp.cuda.Stream.null.synchronize()
    t2 = datetime.now()

    print("Time taken to solve equation: ", t2 - t1)


############################## MULTI-GRID SOLVER ###############################


# The root function of MG-solver. And H is the RHS
def multigrid(H):
    global N
    global vcCnt
    global rConv
    global pAnlt
    global pData, rData

    rData[0] = H
    chMat = cp.zeros(N[0])
    rConv = cp.zeros(vcCnt)

    for i in range(vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        resVal = float(cp.amax(cp.abs(H[1:-1, 1:-1, 1:-1] - chMat)))
        rConv[i] = resVal

        print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

    errVal = float(cp.amax(cp.abs(pAnlt[1:-1, 1:-1, 1:-1] - pData[0][1:-1, 1:-1, 1:-1])))
    print("Error after V-Cycle {0:2d} is {1:4e}\n".format(i+1, errVal))

    return pData[0]


# Multigrid V-cycle without the use of recursion
def v_cycle():
    global VDepth
    global vLev, zeroBC
    global pstSm, preSm

    vLev = 0
    zeroBC = False

    # Pre-smoothing
    smooth(preSm)

    zeroBC = True
    for i in range(VDepth):
        # Compute residual
        calcResidual()

        # Copy smoothed pressure for later use
        sData[vLev] = cp.copy(pData[vLev])

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing!
        if vLev == VDepth:
            solve()
        else:
            smooth(preSm)

    # Prolongation operations
    for i in range(VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Apply homogenous BC so long as we are not at finest mesh (at which vLev = 0)
        if vLev:
            zeroBC = True
        else:
            zeroBC = False

        # Post-smoothing
        smooth(pstSm)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
    global N
    global vLev
    global gsFactor
    global rData, pData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]
    for iCnt in range(sCount):
        imposeBC(pData[vLev])

        # Vectorized Red-Black Gauss-Seidel
        # Update red cells
        # 0, 0, 0 configuration
        pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 1:-1:2] + pData[vLev][:-2:2, 1:-1:2, 1:-1:2]) +
                                               hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, :-2:2, 1:-1:2]) +
                                               hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, :-2:2]) -
                                              hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 1, 1, 0 configuration
        pData[vLev][2::2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 3::2, 1:-1:2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, 2::2, :-2:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 1, 0, 1 configuration
        pData[vLev][2::2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, :-2:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 3::2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 0, 1, 1 configuration
        pData[vLev][1:-1:2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][:-2:2, 2::2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 3::2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 2::2]) * gsFactor[vLev]

        # Update black cells
        # 1, 0, 0 configuration
        pData[vLev][2::2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][2::2, :-2:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][2::2, 1:-1:2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 0, 1, 0 configuration
        pData[vLev][1:-1:2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][:-2:2, 2::2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 0, 0, 1 configuration
        pData[vLev][1:-1:2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][:-2:2, 1:-1:2, 2::2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, :-2:2, 2::2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 3::2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 1, 1, 1 configuration
        pData[vLev][2::2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, 2::2]) +
                                         hzhx[vLev]*(pData[vLev][2::2, 3::2, 2::2] + pData[vLev][2::2, 1:-1:2, 2::2]) +
                                         hxhy[vLev]*(pData[vLev][2::2, 2::2, 3::2] + pData[vLev][2::2, 2::2, 1:-1:2]) -
                                        hxhyhz[vLev]*rData[vLev][2::2, 2::2, 2::2]) * gsFactor[vLev]

    imposeBC(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1, 1:-1] = rData[vLev][1:-1, 1:-1, 1:-1] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    rData[vLev][1:-1, 1:-1, 1:-1] = (iTemp[pLev][1:-1:2, 1:-1:2, 1:-1:2] + iTemp[pLev][2::2, 2::2, 2::2] +
                                     iTemp[pLev][1:-1:2, 1:-1:2, 2::2] + iTemp[pLev][2::2, 2::2, 1:-1:2] +
                                     iTemp[pLev][1:-1:2, 2::2, 1:-1:2] + iTemp[pLev][2::2, 1:-1:2, 2::2] +
                                     iTemp[pLev][2::2, 1:-1:2, 1:-1:2] + iTemp[pLev][1:-1:2, 2::2, 2::2])/8


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global N, vLev
    global gsFactor
    global maxCount
    global tolerance
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]

    jCnt = 0
    while True:
        imposeBC(pData[vLev])

        # Vectorized Red-Black Gauss-Seidel
        # Update red cells
        # 0, 0, 0 configuration
        pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 1:-1:2] + pData[vLev][:-2:2, 1:-1:2, 1:-1:2]) +
                                               hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, :-2:2, 1:-1:2]) +
                                               hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, :-2:2]) -
                                              hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 1, 1, 0 configuration
        pData[vLev][2::2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 1:-1:2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 3::2, 1:-1:2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, 2::2, :-2:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 1, 0, 1 configuration
        pData[vLev][2::2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][2::2, :-2:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 3::2] + pData[vLev][2::2, 1:-1:2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 0, 1, 1 configuration
        pData[vLev][1:-1:2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 2::2] + pData[vLev][:-2:2, 2::2, 2::2]) +
                                           hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 2::2] + pData[vLev][1:-1:2, 1:-1:2, 2::2]) +
                                           hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 3::2] + pData[vLev][1:-1:2, 2::2, 1:-1:2]) -
                                          hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 2::2]) * gsFactor[vLev]

        # Update black cells
        # 1, 0, 0 configuration
        pData[vLev][2::2, 1:-1:2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][3::2, 1:-1:2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][2::2, :-2:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][2::2, 1:-1:2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][2::2, 1:-1:2, 1:-1:2]) * gsFactor[vLev]

        # 0, 1, 0 configuration
        pData[vLev][1:-1:2, 2::2, 1:-1:2] = (hyhz[vLev]*(pData[vLev][2::2, 2::2, 1:-1:2] + pData[vLev][:-2:2, 2::2, 1:-1:2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 3::2, 1:-1:2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, :-2:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 2::2, 1:-1:2]) * gsFactor[vLev]

        # 0, 0, 1 configuration
        pData[vLev][1:-1:2, 1:-1:2, 2::2] = (hyhz[vLev]*(pData[vLev][2::2, 1:-1:2, 2::2] + pData[vLev][:-2:2, 1:-1:2, 2::2]) +
                                             hzhx[vLev]*(pData[vLev][1:-1:2, 2::2, 2::2] + pData[vLev][1:-1:2, :-2:2, 2::2]) +
                                             hxhy[vLev]*(pData[vLev][1:-1:2, 1:-1:2, 3::2] + pData[vLev][1:-1:2, 1:-1:2, 1:-1:2]) -
                                            hxhyhz[vLev]*rData[vLev][1:-1:2, 1:-1:2, 2::2]) * gsFactor[vLev]

        # 1, 1, 1 configuration
        pData[vLev][2::2, 2::2, 2::2] = (hyhz[vLev]*(pData[vLev][3::2, 2::2, 2::2] + pData[vLev][1:-1:2, 2::2, 2::2]) +
                                         hzhx[vLev]*(pData[vLev][2::2, 3::2, 2::2] + pData[vLev][2::2, 1:-1:2, 2::2]) +
                                         hxhy[vLev]*(pData[vLev][2::2, 2::2, 3::2] + pData[vLev][2::2, 2::2, 1:-1:2]) -
                                        hxhyhz[vLev]*rData[vLev][2::2, 2::2, 2::2]) * gsFactor[vLev]

        maxErr = cp.amax(cp.abs(rData[vLev][1:-1, 1:-1, 1:-1] - laplace(pData[vLev])))
        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Gauss-Seidel solver not converging. Aborting")
            quit()

    imposeBC(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev][1:-1:2, 1:-1:2, 1:-1:2] = pData[vLev][2::2, 1:-1:2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 1:-1:2, 2::2] = \
    pData[vLev][2::2, 2::2, 1:-1:2] = pData[vLev][1:-1:2, 2::2, 2::2] = pData[vLev][2::2, 1:-1:2, 2::2] = pData[vLev][2::2, 2::2, 2::2] = pData[pLev][1:-1, 1:-1, 1:-1]


# Computes the 3D laplacian of function
def laplace(function):
    global vLev
    global hx2, hy2, hz2

    laplacian = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1])/hx2[vLev] + 
                 (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1])/hy2[vLev] +
                 (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:])/hz2[vLev])

    return laplacian


############################## BOUNDARY CONDITION ###############################


# The name of this function is self-explanatory. It imposes BC on P
def imposeBC(P):
    global zeroBC
    global pWallX, pWallY, pWallZ

    # Dirichlet BC
    if zeroBC:
        # Homogenous BC
        # Left Wall
        P[0, :, :] = -P[1, :, :]

        # Right Wall
        P[-1, :, :] = -P[-2, :, :]

        # Front Wall
        P[:, 0, :] = -P[:, 1, :]

        # Back Wall
        P[:, -1, :] = -P[:, -2, :]

        # Bottom Wall
        P[:, :, 0] = -P[:, :, 1]

        # Top Wall
        P[:, :, -1] = -P[:, :, -2]

    else:
        # Non-homogenous BC
        # Left Wall
        P[0, :, :] = 2.0*pWallX - P[1, :, :]

        # Right Wall
        P[-1, :, :] = 2.0*pWallX - P[-2, :, :]

        # Front Wall
        P[:, 0, :] = 2.0*pWallY - P[:, 1, :]

        # Back Wall
        P[:, -1, :] = 2.0*pWallY - P[:, -2, :]

        # Bottom Wall
        P[:, :, 0] = 2.0*pWallZ - P[:, :, 1]

        # Top Wall
        P[:, :, -1] = 2.0*pWallZ - P[:, :, -2]


############################### TEST CASE DETAIL ################################


# Calculate the analytical solution and its corresponding Dirichlet BC values
def initDirichlet():
    global N
    global hx, hy, hz
    global pAnlt, pData
    global pWallX, pWallY, pWallZ

    n = N[0]

    # Compute analytical solution, (r^2)/6
    pAnlt = cp.zeros_like(pData[0])

    halfIndX = (n[0] + 1)/2
    halfIndY = (n[1] + 1)/2
    halfIndZ = (n[2] + 1)/2

    xLen, yLen, zLen = 0.5, 0.5, 0.5
    pWallX = cp.zeros_like(pData[0][0, :, :])
    pWallY = cp.zeros_like(pData[0][:, 0, :])
    pWallZ = cp.zeros_like(pData[0][:, :, 0])
    for i in range(n[0] + 2):
        xDist = hx[0]*(i - halfIndX)
        for j in range(n[1] + 2):
            yDist = hy[0]*(j - halfIndY)
            for k in range(n[2] + 2):
                zDist = hz[0]*(k - halfIndZ)
                pAnlt[i, j, k] = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0

                pWallX[j, k] = (xLen*xLen + yDist*yDist + zDist*zDist)/6.0

                pWallY[i, k] = (xDist*xDist + yLen*yLen + zDist*zDist)/6.0

            pWallZ[i, j] = (xDist*xDist + yDist*yDist + zLen*zLen)/6.0


############################## THAT'S IT, FOLKS!! ###############################

if __name__ == '__main__':
    main()

