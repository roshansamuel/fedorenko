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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

############################### GLOBAL VARIABLES ################################

# Choose the grid sizes as indices from below list so that there are 2^n + 3 grid points
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 2 3 5 9 17 33 65 129 257 513 1025 2049 4097 8193 16385
sInd = np.array([5, 5, 5])

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

# N should be of the form 2^n + 1
# Then there will be 2^n + 3 points in total, including 2 ghost points
sLst = [2**x + 1 for x in range(12)]

# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

# Define array of grid spacings along X
hx = [1.0/(x[0]-1) for x in N]

# Define array of grid spacings along Y
hy = [1.0/(x[1]-1) for x in N]

# Define array of grid spacings along Z
hz = [1.0/(x[2]-1) for x in N]

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

    rData = [np.zeros(tuple(x)) for x in nList]
    pData = [np.zeros(tuple(x)) for x in nList + 2]

    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in pData]

    initDirichlet()

    mgRHS = np.ones_like(pData[0])

    # Solve
    mgLHS = multigrid(mgRHS)

    # Normalize solution for Neumann BC
    #mgLHS -= np.mean(mgLHS[1:-1])

    plotResult(2)


############################## MULTI-GRID SOLVER ###############################


# The root function of MG-solver. And H is the RHS
def multigrid(H):
    global N
    global vcCnt
    global rConv
    global pAnlt
    global pData, rData

    rData[0] = H[1:-1, 1:-1, 1:-1]
    chMat = np.zeros(N[0])
    rConv = np.zeros(vcCnt)

    for i in range(vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        resVal = np.amax(np.abs(H[1:-1, 1:-1, 1:-1] - chMat))
        rConv[i] = resVal

        print("Residual after V-Cycle {0:2d} is {1:.4e}".format(i+1, resVal))

    errVal = np.amax(np.abs(pAnlt[1:-1, 1:-1, 1:-1] - pData[0][1:-1, 1:-1, 1:-1]))
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
    global gsFactor
    global rData, pData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]
    for iCnt in range(sCount):
        imposeBC(pData[vLev])

        # Gauss-Seidel smoothing
        for i in range(1, n[0]+1):
            for j in range(1, n[1]+1):
                for k in range(1, n[2]+1):
                    pData[vLev][i, j, k] = (hyhz[vLev]*(pData[vLev][i+1, j, k] + pData[vLev][i-1, j, k]) +
                                            hzhx[vLev]*(pData[vLev][i, j+1, k] + pData[vLev][i, j-1, k]) +
                                            hxhy[vLev]*(pData[vLev][i, j, k+1] + pData[vLev][i, j, k-1]) -
                                          hxhyhz[vLev]*rData[vLev][i-1, j-1, k-1]) * gsFactor[vLev]

    imposeBC(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev][1:-1, 1:-1, 1:-1] = rData[vLev] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n to a smaller array of size 2^(n - 1)
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    n = N[vLev]
    for i in range(1, n[0] + 1):
        i2 = i*2
        for j in range(1, n[1] + 1):
            j2 = j*2
            for k in range(1, n[2] + 1):
                k2 = k*2
                facePoints = (iTemp[pLev][i2, j2 - 1, k2 - 1] + iTemp[pLev][i2 - 2, j2 - 1, k2 - 1] +
                              iTemp[pLev][i2 - 1, j2, k2 - 1] + iTemp[pLev][i2 - 1, j2 - 2, k2 - 1] +
                              iTemp[pLev][i2 - 1, j2 - 1, k2] + iTemp[pLev][i2 - 1, j2 - 1, k2 - 2])*0.0625

                edgePoints = (iTemp[pLev][i2 - 1, j2, k2] + iTemp[pLev][i2 - 1, j2 - 2, k2 - 2] +
                              iTemp[pLev][i2 - 1, j2 - 2, k2] + iTemp[pLev][i2 - 1, j2, k2 - 2] +
                              iTemp[pLev][i2, j2 - 1, k2] + iTemp[pLev][i2 - 2, j2 - 1, k2 - 2] +
                              iTemp[pLev][i2, j2 - 1, k2 - 2] + iTemp[pLev][i2 - 2, j2 - 1, k2] +
                              iTemp[pLev][i2, j2, k2 - 1] + iTemp[pLev][i2 - 2, j2 - 2, k2 - 1] +
                              iTemp[pLev][i2, j2 - 2, k2 - 1] + iTemp[pLev][i2 - 2, j2, k2 - 1])*0.03125

                vertPoints = (iTemp[pLev][i2, j2, k2] + iTemp[pLev][i2 - 2, j2 - 2, k2 - 2] +
                              iTemp[pLev][i2, j2, k2 - 2] + iTemp[pLev][i2 - 2, j2 - 2, k2] +
                              iTemp[pLev][i2, j2 - 2, k2] + iTemp[pLev][i2 - 2, j2, k2 - 2] +
                              iTemp[pLev][i2 - 2, j2, k2] + iTemp[pLev][i2, j2 - 2, k2 - 2])*0.015625

                rData[vLev][i-1, j-1, k-1] = facePoints + edgePoints + vertPoints + iTemp[pLev][i2 - 1, j2 - 1, k2 - 1]*0.125


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global N, vLev
    global gsFactor
    global maxCount
    global tolerance
    global pData, rData
    global hyhz, hzhx, hxhy, hxhyhz

    n = N[vLev]
    solLap = np.zeros(n)

    jCnt = 0
    while True:
        imposeBC(pData[vLev])

        # Gauss-Seidel iterative solver
        for i in range(1, n[0]+1):
            for j in range(1, n[1]+1):
                for k in range(1, n[2]+1):
                    pData[vLev][i, j, k] = (hyhz[vLev]*(pData[vLev][i+1, j, k] + pData[vLev][i-1, j, k]) +
                                            hzhx[vLev]*(pData[vLev][i, j+1, k] + pData[vLev][i, j-1, k]) +
                                            hxhy[vLev]*(pData[vLev][i, j, k+1] + pData[vLev][i, j, k-1]) -
                                          hxhyhz[vLev]*rData[vLev][i-1, j-1, k-1]) * gsFactor[vLev]

        maxErr = np.amax(np.abs(rData[vLev] - laplace(pData[vLev])))
        if maxErr < tolerance:
            break

        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging. Aborting")
            quit()

    imposeBC(pData[vLev])


# Interpolates the data from an array of size 2^n to a larger array of size 2^(n + 1)
def prolong():
    global N
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    pData[vLev].fill(0.0)

    n = N[vLev]
    for i in range(1, n[0] + 1):
        i2 = int(i/2) + 1
        if i % 2:
            for j in range(1, n[1] + 1):
                j2 = int(j/2) + 1
                if j % 2:
                    for k in range(1, n[2] + 1):
                        k2 = int(k/2) + 1
                        if k % 2:
                            pData[vLev][i, j, k] = pData[pLev][i2, j2, k2]
                        else:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2, j2, k2 - 1])/2.0
                else:
                    for k in range(1, n[2] + 1):
                        k2 = int(k/2) + 1
                        if k % 2:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2, j2 - 1, k2])/2.0
                        else:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2, j2 - 1, k2 - 1] +
                                                    pData[pLev][i2, j2 - 1, k2] + pData[pLev][i2, j2, k2 - 1])/4.0
        else:
            for j in range(1, n[1] + 1):
                j2 = int(j/2) + 1
                if j % 2:
                    for k in range(1, n[2] + 1):
                        k2 = int(k/2) + 1
                        if k % 2:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2 - 1, j2, k2])/2.0
                        else:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2 - 1, j2, k2 - 1] +
                                                    pData[pLev][i2 - 1, j2, k2] + pData[pLev][i2, j2, k2 - 1])/4.0
                else:
                    for k in range(1, n[2] + 1):
                        k2 = int(k/2) + 1
                        if k % 2:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2 - 1, j2 - 1, k2] +
                                                    pData[pLev][i2 - 1, j2, k2] + pData[pLev][i2, j2 - 1, k2])/4.0
                        else:
                            pData[vLev][i, j, k] = (pData[pLev][i2, j2, k2] + pData[pLev][i2 - 1, j2 - 1, k2 - 1] +
                                                    pData[pLev][i2 - 1, j2, k2] + pData[pLev][i2, j2 - 1, k2 - 1] + 
                                                    pData[pLev][i2, j2 - 1, k2] + pData[pLev][i2 - 1, j2, k2 - 1] +
                                                    pData[pLev][i2, j2, k2 - 1] + pData[pLev][i2 - 1, j2 - 1, k2])/8.0


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
        P[0, :, :] = -P[2, :, :]

        # Right Wall
        P[-1, :, :] = -P[-3, :, :]

        # Front wall
        P[:, 0, :] = -P[:, 2, :]

        # Back wall
        P[:, -1, :] = -P[:, -3, :]

        # Bottom wall
        P[:, :, 0] = -P[:, :, 2]

        # Top wall
        P[:, :, -1] = -P[:, :, -3]

    else:
        # Non-homogenous BC
        # Left Wall
        P[0, :, :] = 2.0*pWallX - P[2, :, :]

        # Right Wall
        P[-1, :, :] = 2.0*pWallX - P[-3, :, :]

        # Front wall
        P[:, 0, :] = 2.0*pWallY - P[:, 2, :]

        # Back wall
        P[:, -1, :] = 2.0*pWallY - P[:, -3, :]

        # Bottom wall
        P[:, :, 0] = 2.0*pWallZ - P[:, :, 2]

        # Top wall
        P[:, :, -1] = 2.0*pWallZ - P[:, :, -3]


############################### PLOTTING ROUTINE ################################


# plotType = 0: Plot computed and analytic solution together
# plotType = 1: Plot error in computed solution w.r.t. analytic solution
# plotType = 2: Plot convergence of residual against V-Cycles
# Any other value for plotType, and the function will barf.
def plotResult(plotType):
    global N
    global pAnlt
    global pData
    global rConv

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams["font.weight"] = "medium"

    plt.figure(figsize=(13, 9))

    n = N[0]
    xPts = np.linspace(-0.5, 0.5, n[0]+1)

    pSoln = pData[0]
    # Plot the computed solution on top of the analytic solution.
    if plotType == 0:
        plt.plot(xPts, pAnlt, label='Analytic', marker='*', markersize=20, linewidth=4)
        plt.plot(xPts, pSoln[1:-1], label='Computed', marker='+', markersize=20, linewidth=4)
        plt.xlabel('x', fontsize=40)
        plt.ylabel('p', fontsize=40)

    # Plot the error in computed solution with respect to analytic solution.
    elif plotType == 1:
        pErr = np.abs(pAnlt - pSoln[1:-1])
        plt.semilogy(xPts, pErr, label='Error', marker='*', markersize=20, linewidth=4)
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


############################### TEST CASE DETAIL ################################


# Calculate the analytical solution and its corresponding Dirichlet BC values
def initDirichlet():
    global N
    global hx, hy, hz
    global pAnlt, pData
    global pWallX, pWallY, pWallZ

    n = N[0]

    # Compute analytical solution, (r^2)/6
    pAnlt = np.zeros_like(pData[0])

    halfIndX = int(n[0]/2) + 1
    halfIndY = int(n[1]/2) + 1
    halfIndZ = int(n[2]/2) + 1

    for i in range(n[0] + 2):
        xDist = hx[0]*(i - halfIndX)
        for j in range(n[1] + 2):
            yDist = hy[0]*(j - halfIndY)
            for k in range(n[2] + 2):
                zDist = hz[0]*(k - halfIndZ)
                pAnlt[i, j, k] = (xDist*xDist + yDist*yDist + zDist*zDist)/6.0

    # Value of P at walls according to analytical solution
    pWallX = pAnlt[1, :, :]
    pWallY = pAnlt[:, 1, :]
    pWallZ = pAnlt[:, :, 1]


############################## THAT'S IT, FOLKS!! ###############################

if __name__ == '__main__':
    main()

