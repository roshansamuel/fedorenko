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
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

############################### GLOBAL VARIABLES ################################

# Choose the grid sizes as indices from below list so that there are 2^n + 2 grid points
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = np.array([6, 6])

# Flag to switch between uniform and non-uniform grid with tan-hyp stretching
nuFlag = True

# Stretching parameter for tangent-hyperbolic grid
beta = 1.0

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
N = [(sLst[x[0]], sLst[x[1]]) for x in [sInd - y for y in range(VDepth + 1)]]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*N[-1][0]*N[-1][1]

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

    rData = [np.zeros(tuple(x)) for x in nList]
    pData = [np.zeros(tuple(x)) for x in nList + 2]

    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in rData]

    initGrid()
    initDirichlet()

    mgRHS = np.ones_like(pData[0])

    # Solve
    t1 = datetime.now()
    mgLHS = multigrid(mgRHS)
    t2 = datetime.now()

    print("Time taken to solve equation: ", t2 - t1)

    plotResult(0)


############################## MULTI-GRID SOLVER ###############################


# The root function of MG-solver. And H is the RHS
def multigrid(H):
    global N
    global vcCnt
    global rConv
    global pAnlt
    global pData, rData

    rData[0] = H[1:-1, 1:-1]
    chMat = np.zeros(N[0])
    rConv = np.zeros(vcCnt)

    for i in range(vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        resVal = np.amax(np.abs(H[1:-1, 1:-1] - chMat))
        rConv[i] = resVal

        print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

    errVal = np.amax(np.abs(pAnlt - pData[0][1:-1, 1:-1]))
    print("Error after V-Cycle {0:2d} is {1:.4e}\n".format(i+1, errVal))

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
        sData[vLev] = np.copy(pData[vLev])

        # Restrict to coarser level
        restrict()

        # Reinitialize pressure at coarser level to 0 - this is critical!
        pData[vLev].fill(0.0)

        # If the coarsest level is reached, solve. Otherwise, keep smoothing!
        if vLev == VDepth:
            solve()
            #smooth(preSm)
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
    global nuFlag
    global rData, pData
    global ihx2, i2hx, ihz2, i2hz
    global xixx, xix2, ztzz, ztz2

    n = N[vLev]
    for iCnt in range(sCount):
        imposeBC(pData[vLev])

        # Gauss-Seidel smoothing
        if nuFlag:
            # For non-uniform grid
            for i in range(1, n[0]+1):
                for j in range(1, n[1]+1):
                    pData[vLev][i, j] = (xix2[vLev][i-1] * ihx2[vLev] * (pData[vLev][i+1, j] + pData[vLev][i-1, j]) +
                                         xixx[vLev][i-1] * i2hx[vLev] * (pData[vLev][i+1, j] - pData[vLev][i-1, j]) +
                                         ztz2[vLev][j-1] * ihz2[vLev] * (pData[vLev][i, j+1] + pData[vLev][i, j-1]) +
                                         ztzz[vLev][j-1] * i2hz[vLev] * (pData[vLev][i, j+1] - pData[vLev][i, j-1]) -
                                        rData[vLev][i-1, j-1]) / (2.0*(ihx2[vLev]*xix2[vLev][i-1] +
                                                                       ihz2[vLev]*ztz2[vLev][j-1]))
        else:
            # For uniform grid
            for i in range(1, n[0]+1):
                for j in range(1, n[1]+1):
                    pData[vLev][i, j] = (ihx2[vLev] * (pData[vLev][i+1, j] + pData[vLev][i-1, j]) +
                                         ihz2[vLev] * (pData[vLev][i, j+1] + pData[vLev][i, j-1]) -
                                        rData[vLev][i-1, j-1]) / (2.0*(ihx2[vLev] + ihz2[vLev]))

    imposeBC(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev] = rData[vLev] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    '''
    rData[vLev] = (iTemp[pLev][::2, ::2, ::2] + iTemp[pLev][1::2, 1::2, 1::2] +
                   iTemp[pLev][::2, ::2, 1::2] + iTemp[pLev][1::2, 1::2, ::2] +
                   iTemp[pLev][::2, 1::2, ::2] + iTemp[pLev][1::2, ::2, 1::2] +
                   iTemp[pLev][1::2, ::2, ::2] + iTemp[pLev][::2, 1::2, 1::2])/8
    '''
    n = N[vLev]
    for i in range(n[0]):
        i2 = i*2
        for k in range(n[1]):
            k2 = k*2
            rData[vLev][i, k] = 0.25*(iTemp[pLev][i2 + 1, k2 + 1] + iTemp[pLev][i2, k2 + 1] + iTemp[pLev][i2 + 1, k2] + iTemp[pLev][i2, k2])


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global N, vLev
    global nuFlag
    global maxCount
    global tolerance
    global pData, rData
    global ihx2, i2hx, ihz2, i2hz
    global xixx, xix2, ztzz, ztz2

    n = N[vLev]
    solLap = np.zeros(n)

    jCnt = 0
    while True:
        imposeBC(pData[vLev])

        # Gauss-Seidel iterative solver
        if nuFlag:
            # For non-uniform grid
            for i in range(1, n[0]+1):
                for j in range(1, n[1]+1):
                    pData[vLev][i, j] = (xix2[vLev][i-1] * ihx2[vLev] * (pData[vLev][i+1, j] + pData[vLev][i-1, j]) +
                                         xixx[vLev][i-1] * i2hx[vLev] * (pData[vLev][i+1, j] - pData[vLev][i-1, j]) +
                                         ztz2[vLev][j-1] * ihz2[vLev] * (pData[vLev][i, j+1] + pData[vLev][i, j-1]) +
                                         ztzz[vLev][j-1] * i2hz[vLev] * (pData[vLev][i, j+1] - pData[vLev][i, j-1]) -
                                        rData[vLev][i-1, j-1]) / (2.0*(ihx2[vLev]*xix2[vLev][i-1] +
                                                                       ihz2[vLev]*ztz2[vLev][j-1]))
        else:
            # For uniform grid
            for i in range(1, n[0]+1):
                for j in range(1, n[1]+1):
                    pData[vLev][i, j] = (ihx2[vLev] * (pData[vLev][i+1, j] + pData[vLev][i-1, j]) +
                                         ihz2[vLev] * (pData[vLev][i, j+1] + pData[vLev][i, j-1]) -
                                        rData[vLev][i-1, j-1]) / (2.0*(ihx2[vLev] + ihz2[vLev]))

        maxErr = np.amax(np.abs(rData[vLev] - laplace(pData[vLev])))
        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("WARNING: Iterative solver not converging at coarsest level")
            break

    imposeBC(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global N
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    n = N[vLev]
    for i in range(1, n[0] + 1):
        i2 = int((i-1)/2) + 1
        for k in range(1, n[1] + 1):
            k2 = int((k-1)/2) + 1
            pData[vLev][i, k] = pData[pLev][i2, k2]


# Computes the 2D laplacian of function
def laplace(function):
    global vLev
    global nuFlag
    global ihx2, i2hx, ihz2, i2hz
    global xixx, xix2, ztzz, ztz2

    if nuFlag:
        # For non-uniform grid
        laplacian = (xix2[vLev] * ihx2[vLev] * (function[2:, 1:-1] - 2.0*function[1:-1, 1:-1] + function[:-2, 1:-1]) + \
                     xixx[vLev] * i2hx[vLev] * (function[2:, 1:-1] - function[:-2, 1:-1]) +
                     ztz2[vLev] * ihz2[vLev] * (function[1:-1, 2:] - 2.0*function[1:-1, 1:-1] + function[1:-1, :-2]) + \
                     ztzz[vLev] * i2hz[vLev] * (function[1:-1, 2:] - function[1:-1, :-2]))
    else:
        # For uniform grid
        laplacian = ((function[:-2, 1:-1] - 2.0*function[1:-1, 1:-1] + function[2:, 1:-1]) * ihx2[vLev] + 
                     (function[1:-1, :-2] - 2.0*function[1:-1, 1:-1] + function[1:-1, 2:]) * ihz2[vLev])

    return laplacian


############################## BOUNDARY CONDITION ###############################


# The name of this function is self-explanatory. It imposes BC on P
def imposeBC(P):
    global zeroBC
    global pWallX, pWallZ

    # Dirichlet BC
    if zeroBC:
        # Homogenous BC
        # Left Wall
        P[0, :] = -P[1, :]

        # Right Wall
        P[-1, :] = -P[-2, :]

        # Bottom Wall
        P[:, 0] = -P[:, 1]

        # Top Wall
        P[:, -1] = -P[:, -2]

    else:
        # Non-homogenous BC
        # Left Wall
        P[0, :] = 2.0*pWallX - P[1, :]

        # Right Wall
        P[-1, :] = 2.0*pWallX - P[-2, :]

        # Bottom Wall
        P[:, 0] = 2.0*pWallZ - P[:, 1]

        # Top Wall
        P[:, -1] = 2.0*pWallZ - P[:, -2]


############################## GRID INITIALIZATION ##############################


# Initialize the grid. This is relevant only for non-uniform grids
def initGrid():
    global N
    global nuFlag
    global hx, xPts, i2hx, ihx2, xixx, xix2
    global hz, zPts, i2hz, ihz2, ztzz, ztz2

    hx0 = 1.0/(N[0][0])
    hz0 = 1.0/(N[0][1])

    hx = np.zeros(VDepth+1)
    hz = np.zeros(VDepth+1)

    ihx2 = np.zeros(VDepth+1)
    ihz2 = np.zeros(VDepth+1)

    i2hx = np.zeros(VDepth+1)
    i2hz = np.zeros(VDepth+1)

    for i in range(VDepth+1):
        hx[i] = hx0*(2**i)
        hz[i] = hz0*(2**i)

        ihx2[i] = 1.0/(hx[i]*hx[i])
        ihz2[i] = 1.0/(hz[i]*hz[i])

        i2hx[i] = 1.0/(2.0*hx[i])
        i2hz[i] = 1.0/(2.0*hz[i])

    # Uniform grid default values
    vPts = [np.linspace(-0.5, 0.5, n[0]+1) for n in N]
    xPts = [(x[1:] + x[:-1])/2.0 for x in vPts]
    xi_x = [np.ones_like(i) for i in xPts]
    xix2 = [np.ones_like(i) for i in xPts]
    xixx = [np.zeros_like(i) for i in xPts]

    vPts = [np.linspace(-0.5, 0.5, n[1]+1) for n in N]
    zPts = [(z[1:] + z[:-1])/2.0 for z in vPts]
    zt_z = [np.ones_like(i) for i in zPts]
    ztz2 = [np.ones_like(i) for i in zPts]
    ztzz = [np.zeros_like(i) for i in zPts]

    # Overwrite above arrays with values for tangent-hyperbolic grid is nuFlag is enabled.
    if nuFlag:
        for i in range(VDepth+1):
            n = N[i]

            vPts = np.linspace(0.0, 1.0, n[0]+1)
            xi = (vPts[1:] + vPts[:-1])/2.0
            xPts[i] = np.array([(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in xi])
            xi_x[i] = np.array([np.tanh(beta)/(beta*(1.0 - ((1.0 - 2.0*k)*np.tanh(beta))**2.0)) for k in xPts[i]])
            xixx[i] = np.array([-4.0*(np.tanh(beta)**3.0)*(1.0 - 2.0*k)/(beta*(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0)**2.0) for k in xPts[i]])
            xix2[i] = np.array([k*k for k in xi_x[i]])
            xPts[i] -= 0.5

            vPts = np.linspace(0.0, 1.0, n[1]+1)
            zt = (vPts[1:] + vPts[:-1])/2.0
            zPts[i] = np.array([(1.0 - np.tanh(beta*(1.0 - 2.0*i))/np.tanh(beta))/2.0 for i in zt])
            zt_z[i] = np.array([np.tanh(beta)/(beta*(1.0 - ((1.0 - 2.0*k)*np.tanh(beta))**2.0)) for k in zPts[i]])
            ztzz[i] = np.array([-4.0*(np.tanh(beta)**3.0)*(1.0 - 2.0*k)/(beta*(1.0 - (np.tanh(beta)*(1.0 - 2.0*k))**2.0)**2.0) for k in zPts[i]])
            ztz2[i] = np.array([k*k for k in zt_z[i]])
            zPts[i] -= 0.5

    # Reshape arrays to make it easier to multiply with 3D arrays
    xixx = [x[:, np.newaxis] for x in xixx]
    xix2 = [x[:, np.newaxis] for x in xix2]


############################### TEST CASE DETAIL ################################


# Calculate the analytical solution and its corresponding Dirichlet BC values
def initDirichlet():
    global N
    global xPts, zPts
    global pAnlt, pData
    global pWallX, pWallZ

    xLen, zLen = 0.5, 0.5

    # Compute analytical solution, (r^2)/4
    pAnlt = np.zeros(N[0])
    x = xPts[0]
    z = zPts[0]

    npax = np.newaxis
    pAnlt = (x[:, npax]**2 + z[:]**2)/4.0

    pWallX = np.zeros_like(pData[0][0, :])
    pWallZ = np.zeros_like(pData[0][:, 0])

    pWallX[1:-1] = (xLen**2 + z[:]**2)/4.0
    pWallZ[1:-1] = (x[:]**2 + zLen**2)/4.0


############################### PLOTTING ROUTINE ################################


# plotType = 0: Plot computed and analytic solution together
# plotType = 1: Plot error in computed solution w.r.t. analytic solution
# plotType = 2: Plot convergence of residual against V-Cycles
# Any other value for plotType, and the function will barf.
def plotResult(plotType):
    global N
    global zPts
    global pAnlt
    global pData
    global rConv

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams["font.weight"] = "medium"

    plt.figure(figsize=(13, 9))

    n = N[0]
    xMid = int(n[0]/2)

    pSoln = pData[0]
    # Plot the computed solution on top of the analytic solution.
    if plotType == 0:
        plt.plot(zPts[0], pAnlt[xMid, :], label='Analytic', marker='*', markersize=20, linewidth=4)
        plt.plot(zPts[0], pSoln[xMid, 1:-1], label='Computed', marker='+', markersize=20, linewidth=4)
        plt.xlabel('x', fontsize=40)
        plt.ylabel('p', fontsize=40)

    # Plot the error in computed solution with respect to analytic solution.
    elif plotType == 1:
        pErr = np.abs(pAnlt - pSoln[1:-1, 1:-1])
        plt.semilogy(zPts[0], pErr[xMid, :], label='Error', marker='*', markersize=20, linewidth=4)
        plt.xlabel('x', fontsize=40)
        plt.ylabel('e_p', fontsize=40)

    # Plot the convergence of residual
    elif plotType == 2:
        vcAxis = np.arange(len(rConv)) + 1
        plt.semilogy(vcAxis, rConv, label='Residual', marker='*', markersize=20, linewidth=4)
        plt.xlabel('V-Cycles', fontsize=40)
        plt.ylabel('Residual', fontsize=40)

    axes = plt.gca()
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=40)
    plt.show()


############################## THAT'S IT, FOLKS!! ###############################

if __name__ == '__main__':
    main()

