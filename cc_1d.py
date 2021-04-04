# Import all necessary modules
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

############################### GLOBAL VARIABLES ################################

# Choose grid size as an index from below list
# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
sInd = 9

# Depth of each V-cycle in multigrid (ideally VDepth = sInd - 1)
VDepth = sInd - 1

# Number of V-cycles to be computed
vcCnt = 10

# Number of iterations during pre-smoothing
preSm = 2

# Number of iterations during post-smoothing
pstSm = 2

# Tolerance value for iterative solver
tolerance = 1.0e-6

# N should be of the form 2^n
# Then there will be 2^n + 2 points in total, including 2 ghost points
sLst = [2**x for x in range(15)]

# Get array of grid sizes corresponding to each level of V-Cycle
N = sLst[sInd:sInd - VDepth - 1:-1]

# Define array of grid spacings
hx0 = 1.0/N[0]
hx = [hx0*(2**x) for x in range(VDepth+1)]

# Square of hx, used in finite difference formulae
hx2 = [x*x for x in hx]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*sLst[sInd]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0

##################################### MAIN ######################################

def main():
    global N
    global xPts
    global pData, pAnlt
    global rData, sData, iTemp

    nList = np.array(N)

    rData = [np.zeros(x) for x in nList]
    pData = [np.zeros(x) for x in nList + 2]

    sData = [np.zeros_like(x) for x in pData]
    iTemp = [np.zeros_like(x) for x in rData]

    xPts = np.linspace(-0.5, 0.5, N[0])

    mgRHS = np.zeros(N[0] + 2)

    # RHS is merely x over -0.5 to 0.5
    mgRHS[1:-1] = xPts

    # Solve
    mgLHS = multigrid(mgRHS)

    # Normalize solution for Neumann BC
    mgLHS -= np.mean(mgLHS[1:-1])

    # Analytical solution
    pAnlt = xPts*xPts*xPts/6.0 - xPts/8.0

    solError = np.max(np.abs(pAnlt - mgLHS[1:-1]))

    print("Error in computed solution is {0:.4e}".format(solError))

    plotResult(2)


############################## MULTI-GRID SOLVER ###############################


# The root function of MG-solver. And H is the RHS
def multigrid(H):
    global N
    global vcCnt
    global rConv
    global pData, rData

    rData[0] = H[1:-1]
    chMat = np.zeros(N[0])
    rConv = np.zeros(vcCnt)

    for i in range(vcCnt):
        v_cycle()

        chMat = laplace(pData[0])
        resVal = np.amax(np.abs(H[1:-1] - chMat))
        rConv[i] = resVal

        print("Residual after V-Cycle {0:2d} is {1:.4e}\n".format(i+1, resVal))

    return pData[0]


# Multigrid V-cycle without the use of recursion
def v_cycle():
    global vLev
    global VDepth
    global pstSm, preSm

    vLev = 0

    # Pre-smoothing
    smooth(preSm)

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
            #solve()
            smooth(preSm)
        else:
            smooth(preSm)

    # Prolongation operations
    for i in range(VDepth):
        # Prolong pressure to next finer level
        prolong()

        # Add previously stored smoothed data
        pData[vLev] += sData[vLev]

        # Post-smoothing
        smooth(pstSm)


# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
    global N
    global hx2
    global vLev
    global rData, pData

    n = N[vLev]
    for i in range(sCount):
        imposeBC(pData[vLev])

        # Gauss-Seidel smoothing
        for j in range(1, n+1):
            pData[vLev][j] = (pData[vLev][j+1] + pData[vLev][j-1] - hx2[vLev]*rData[vLev][j-1])*0.5

    imposeBC(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
    global vLev
    global iTemp, rData, pData

    iTemp[vLev].fill(0.0)
    iTemp[vLev] = rData[vLev] - laplace(pData[vLev])


# Restricts the data from an array of size 2^n + 1 to a smaller array of size 2^(n - 1) + 1
def restrict():
    global N
    global vLev
    global iTemp, rData

    pLev = vLev
    vLev += 1

    for i in range(N[vLev]):
        i2 = i*2
        rData[vLev][i] = 0.5*(iTemp[pLev][i2 + 1] + iTemp[pLev][i2])


# Solves at coarsest level using the Gauss-Seidel iterative solver
def solve():
    global vLev
    global N, hx2
    global maxCount
    global tolerance
    global pData, rData

    n = N[vLev]
    solLap = np.zeros(n)

    jCnt = 0
    while True:
        #imposeBC(pData[vLev])

        # Gauss-Seidel iterative solver
        for i in range(1, n+1):
            pData[vLev][i] = (pData[vLev][i+1] + pData[vLev][i-1] - hx2[vLev]*rData[vLev][i-1])*0.5

        #imposeBC(pData[vLev])

        lData = (pData[vLev][2:] - 2.0*pData[vLev][1:-1] + pData[vLev][:-2])/hx2[vLev]
        maxErr = np.amax(np.abs(rData[vLev] - lData))
        if maxErr < tolerance:
            print(jCnt, maxErr)
            break

        jCnt += 1
        if jCnt > maxCount:
            print("Iterative solver not converging.\n")
            exit()

    imposeBC(pData[vLev])

    return 0


# Interpolates the data from an array of size 2^n + 1 to a larger array of size 2^(n + 1) + 1
def prolong():
    global N
    global vLev
    global pData

    pLev = vLev
    vLev -= 1

    for i in range(1, N[vLev] + 1):
        i2 = int((i - 1)/2) + 1;
        pData[vLev][i] = pData[pLev][i2]


# Computes the 1D laplacian of function
def laplace(function):
    global vLev
    global N, hx2

    n = N[vLev]

    laplacian = np.zeros(n)
    laplacian = (function[2:] - 2.0*function[1:-1] + function[:-2]) / hx2[vLev]

    return laplacian


############################## BOUNDARY CONDITION ###############################


# The name of this function is self-explanatory. It imposes BC on P
def imposeBC(P):
    # Neumann BC
    P[0] = P[1]
    P[-1] = P[-2]


############################### PLOTTING ROUTINE ################################


# plotType = 0: Plot computed and analytic solution together
# plotType = 1: Plot error in computed solution w.r.t. analytic solution
# plotType = 2: Plot convergence of residual against V-Cycles
# Any other value for plotType, and the function will barf.
def plotResult(plotType):
    global N
    global xPts
    global pAnlt
    global pData
    global rConv

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams["font.weight"] = "medium"

    plt.figure(figsize=(13, 9))

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


################################ END OF PROGRAM #################################

if __name__ == '__main__':
    main()

