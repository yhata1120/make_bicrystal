"""
This is the file including mathematic calculation for CSL GBs

The code is mainly from Haddian and some is modified by Yaoshu
The part non-related to my application is deleted. For a complete version of Haddian refer
https://github.com/oekosheri/GB_code
"""

import sys
import random
from math import degrees, atan, sqrt, pi, ceil, cos, acos, sin, gcd, radians
import numpy as np
from numpy import dot, cross
from numpy.linalg import det, norm, inv
from fractions import Fraction
import sympy

def getFZ(M,axis):
    x = np.arange(-50, 50 + 1, 1)
    y = x
    z = x
    tol = 1e-13
    basis = Basis('fcc')
    basis = np.delete(basis,0,axis=0)
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)] 
    indice = np.array(indice, dtype = np.float64)   
    CSL = dot(indice_0, M.T)
    CSL = CSL[np.argsort(norm(CSL, axis=1))]
    dots = abs(dot(CSL, axis.T))
    index = np.where(dots < tol)[0]
    pvs = CSL[index]
    v1 = pvs[0]
    for i in pvs:
        if norm(cross(i,v1)) > tol:
            v2 = i
            break
    if ang(cross(v1, v2), axis) < 0:
        axis = -axis
    return np.vstack((v1,v2,axis)).T
    
def searchCbyR(R,axis):
    """
    --- by Yaoshu ---
    find CSL of the conventional lattice by rotation matrix R and axis
    arguments:
    R -- rotation matrix of CSL
    """
    M = np.zeros([3,3])
    #meshes
    lim = 30
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    tol = 0.0001
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]    
    
    Lattice1 = indice_0
    Lattice2 = np.around(dot(R, Lattice1.T).T, 8)        

    L2f = Lattice2
    index = np.where(np.all(abs(np.round(L2f) - L2f) < 1e-7, axis = 1))[0]

    nn = Lattice2[index]
    TestVecs = nn[np.argsort(norm(nn, axis=1))]
    Found = False
    count = 0

    M[:,2] = axis

    for i in range(len(TestVecs)):
        if abs(dot(TestVecs[i], M[:,2])) < tol:
            M[:,1] = TestVecs[i]
            for j in range(len(TestVecs)):
                if (1 - ang(TestVecs[j], M[:,2]) > tol) and (1 - ang(TestVecs[j], M[:,1]) > tol) and dot(TestVecs[j],M[:,1]) > 0:
                    if (ang(TestVecs[j], cross(M[:,2], M[:,1])) > tol):
                        M[:,0] = TestVecs[j]

                        #confirm right handed
                        if ang(cross(M[:,0], M[:,1]), M[:,2]) < 0:
                            M[:,2] = - M[:,2]
                        Found = True
                        break
            if Found:
                break
                
            if not Found:
                for j in range(len(TestVecs)):
                    if (1 - ang(TestVecs[j], M[:,2]) > tol) and (1 - ang(TestVecs[j], M[:,1]) > tol):
                            if (ang(TestVecs[j], cross(M[:,2], M[:,1])) > tol):
                                M[:,0] = TestVecs[j]

                                #confirm right handed
                                if ang(cross(M[:,0], M[:,1]), M[:,2]) < 0:
                                    M[:,2] = - M[:,2]
                                Found = True
                                break
            if Found:
                break
    if Found:
        return M
    else:
        return None
        
def get_cubic_sigma(uvw, m, n=1):
    """
    ---by Haddian---
    CSL analytical formula based on the book: 'Interfaces in crystalline materials',
     Sutton and Balluffi, clarendon press, 1996.
     generates possible sigma values.
    arguments:
    uvw -- the axis
    m,n -- two integers (n by default 1)
    """
    u, v, w = uvw
    sqsum = u*u + v*v + w*w
    sigma = m*m + n*n * sqsum
    while sigma != 0 and sigma % 2 == 0:
        sigma /= 2
    return sigma if sigma > 1 else None


def get_cubic_theta(uvw, m, n=1):
    """
    ---by Haddian---
    generates possible theta values.
    arguments:
    uvw -- the axis
    m,n -- two integers (n by default 1)
    """
    u, v, w = uvw
    sqsum = u*u + v*v + w*w
    if m > 0:
        return 2 * atan(sqrt(sqsum) * n / m)
    else:
        return pi


def get_theta_m_n_list(uvw, sigma):
    """
    ---by Haddian---
    Finds integers m and n lists that match the input sigma.
    """
    if sigma == 1:
        return [(0., 0., 0.)]
    thetas = []
    max_m = int(ceil(sqrt(4*sigma)))

    for m in range(1, max_m):
        for n in range(1, max_m):
            if gcd(m, n) == 1:
                s = get_cubic_sigma(uvw, m, n)
            if s == sigma:
                theta = (get_cubic_theta(uvw, m, n))
                thetas.append((theta, m, n))
                thetas.sort(key=lambda x: x[0])
    return thetas

def print_list(uvw, limit):
    """
    ---by Haddian---
    prints a list of smallest sigmas/angles for a given axis(uvw).
    """


    for i in range(limit):
        tt = get_theta_m_n_list(uvw, i)
        if len(tt) > 0:
            theta, m, n = tt[0]
            print("Sigma:   {0:3d}  Theta:  {1:5.2f} "
                  .format(i, degrees(theta)))

def getsigmas(uvw, limit):
    """
    ---by Haddian---
    prints a list of smallest sigmas/angles for a given axis(uvw).
    """
    sigmas = []
    thetas = []
    for i in range(limit):

        tt = get_theta_m_n_list(uvw, i)
        if len(tt) > 0 and i > 1:
            theta, m, n = tt[0]
            if (uvw != [1,1,1,]):
                sigmas.append(i)
                thetas.append(degrees(theta))                
            elif ((i%3 != 0) & (uvw ==[1,1,1,])) or (sympy.isprime(i)):
                sigmas.append(i)
                thetas.append(degrees(theta))
    return sigmas, thetas

def rot(a, Theta):
    """
    ---by Haddian---
    produces a rotation matrix.
    arguments:
    a -- an axis
    Theta -- an angle
    """
    c = float(cos(Theta))
    s = float(sin(Theta))
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s,
                      ax * az * (1 - c) + ay * s],
                    [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c),
                        ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s,
                      c + az * az * (1 - c)]], dtype = np.float64)


# Helpful Functions:
#-------------------#

def integer_array(A, tol=1e-7):
    """
    ---by Haddian---
    returns True if an array is ineteger.
    """
    return np.all(abs(np.round(A) - A) < tol)

def angv(a, b):
    """
    ---by Haddian---
    returns the angle between two vectors.
    """
    ang = acos(np.round(dot(a, b)/norm(a)/norm(b), 8))
    return round(degrees(ang), 7)

def ang(a, b):
    """
    ---by Haddian---
    returns the cos(angle) between two vectors.
    """
    ang = np.round(dot(a, b)/norm(a)/norm(b), 7)
    return abs(ang)

def CommonDivisor(a):
    """
    ---by Haddian---
    returns the common factor of vector a and the reduced vector.
    """
    CommFac = []
    a = np.array(a)
    for i in range(2, 10):
        while (a[0] % i == 0 and a[1] % i == 0 and a[2] % i == 0):
            a = a / i
            CommFac.append(i)
    return(a.astype(int), (abs(np.prod(CommFac))))

def SmallestInteger(a):
    """
    ---by Haddian---
    returns the smallest multiple integer to make an integer array.
    """
    a = np.array(a)
    for i in range(1, 200):
        testV = i * a
        if integer_array(testV):
            break
    return (testV, i) if integer_array(testV) else None

def integerMatrix(a):
    """
    ---by Haddian---
    returns an integer matrix from row vectors.
    """
    Found = True
    b = np.zeros((3,3))
    a = np.array(a)
    for i in range(3):
        for j in range(1, 2000):
            testV = j * a[i]
            if integer_array(testV):
                b[i] = testV
                break
        if all(b[i] == 0):
            Found = False
            print("Can not make integer matrix!")
    return (b) if Found else None

def SymmEquivalent(arr):
    """
    ---by Haddian---
    returns cubic symmetric eqivalents of the given 2 dimensional vector.
    """
    Sym = np.zeros([24, 3, 3])
    Sym[0, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Sym[1, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    Sym[2, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    Sym[3, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    Sym[4, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    Sym[5, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    Sym[6, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    Sym[7, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    Sym[8, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    Sym[9, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    Sym[10, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    Sym[11, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    Sym[12, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    Sym[13, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    Sym[14, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]
    Sym[15, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    Sym[16, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    Sym[17, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    Sym[18, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]
    Sym[19, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    Sym[20, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    Sym[21, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    Sym[22, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
    Sym[23, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]

    arr = np.atleast_2d(arr)
    Result = []
    for i in range(len(Sym)):
        for j in range(len(arr)):
            Result.append(dot(Sym[i, :], arr[j]))
    Result = np.array(Result)
    return np.unique(Result, axis=0)

        
def Tilt_Twist_comp(v1, uvw, m, n):
    """
    ---by Haddian---
    returns the tilt and twist components of a given GB plane.
    arguments:
    v1 -- given gb plane
    uvw -- axis of rotation
    m,n -- the two necessary integers
    """
    theta = get_cubic_theta(uvw, m, n)
    R = rot(uvw, theta)
    v2 = np.round(dot(R, v1), 6).astype(int)
    tilt = angv(v1, v2)
    if abs(tilt - degrees(theta)) < 10e-5:
        print("Pure tilt boundary with a tilt component: {0:6.2f}"
              .format(tilt))
    else:
        twist = 2 * acos(cos(theta / 2) / cos(radians(tilt / 2)))
        print("Tilt component: {0:<6.2f} Twist component: {1:6.2f}"
              .format(tilt, twist))


def Create_Possible_GB_Plane_List(uvw, m=5, n=1, lim=5):
    """
    ---by Haddian---
    generates GB planes and specifies the character.

    arguments:
    uvw -- axis of rotation.
    m,n -- the two necessary integers
    lim -- upper limit for the plane indices

    """
    uvw = np.array(uvw)
    Theta = get_cubic_theta(uvw, m, n)
    Sigma = get_cubic_sigma(uvw, m, n)
    R1 = rot(uvw, Theta)

    # List and character of possible GB planes:
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = indice_0[np.argsort(norm(indice_0, axis=1))]

    # extract the minimal cell:
    Min_1, Min_2 = Create_minimal_cell_Method_1(Sigma, uvw, R1)
    V1 = np.zeros([len(indice_0), 3])
    V2 = np.zeros([len(indice_0), 3])
    GBtype = []
    tol = 0.001
    # Mirrorplanes cubic symmetry
    MP = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 1, 0],
                   [0, 1, 1],
                   [1, 0, 1],
                  ], dtype='float')
    # Find GB plane coordinates:
    for i in range(len(indice_0)):
        if CommonDivisor(indice_0[i])[1] <= 1:
            V1[i, :] = (indice_0[i, 0] * Min_1[:, 0] +
                        indice_0[i, 1] * Min_1[:, 1] +
                        indice_0[i, 2] * Min_1[:, 2])
            V2[i, :] = (indice_0[i, 0] * Min_2[:, 0] +
                        indice_0[i, 1] * Min_2[:, 1] +
                        indice_0[i, 2] * Min_2[:, 2])

    V1 = (V1[~np.all(V1 == 0, axis=1)]).astype(int)
    V2 = (V2[~np.all(V2 == 0, axis=1)]).astype(int)
    MeanPlanes = (V1 + V2) / 2

    # Check the type of GB plane: Symmetric tilt, tilt, twist
    for i in range(len(V1)):
        if ang(V1[i], uvw) < tol:

            for j in range(len(SymmEquivalent(MP))):
                if 1 - ang(MeanPlanes[i], SymmEquivalent(MP)[j]) < tol:
                    GBtype.append('Symmetric Tilt')
                    break
            else:
                GBtype.append('Tilt')
        elif 1 - ang(V1[i], uvw) < tol:
            GBtype.append('Twist')
        else:
            GBtype.append('Mixed')

    return (V1, V2, MeanPlanes, GBtype)
    
def Create_minimal_cell_Method_1(sigma, uvw, R):
    """
    ---by Haddian---
    finds Minimal cell by means of a numerical search.
    (An alternative analytical method can be used too).
    arguments:
    sigma -- gb sigma
    uvw -- rotation axis
    R -- rotation matrix
    """
    uvw = np.array(uvw)
    MiniCell_1 = np.zeros([3, 3])
    MiniCell_1[:, 2] = uvw

    lim = 20
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    
    # remove 0 vectors and uvw from the list
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    condition1 = ((abs(dot(indice_0, uvw) / norm(indice_0, axis=1) /
                       norm(uvw))).round(7))
    indice_0 = indice_0[np.where(condition1 != 1)]

    if MiniCell_search(indice_0, MiniCell_1, R, sigma):

        M1, M2 = MiniCell_search(indice_0, MiniCell_1, R, sigma)
        return (M1, M2)
    else:
        return None

def searchcnid(R,gb):
    """
    ---by Yaoshu---
    finds c.n.i.d by the rotation matrix and 
    GB plane normal.For more details of c.n.i.d,
    refer the book: 'Interfaces in crystalline materials',
     Sutton and Balluffi, clarendon press, 1996.
    arguments:
    R -- rotation matrix of the CSL
    gb -- GB plane normal
    """
    #mesh
    lim = 100
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    tol = 1e-10
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)] 
    
    indice_0 = np.array(indice_0,dtype = np.float64)
    #get the lattice points of two grains
    basis1 = np.array([[0.5, 0, 0.5],
                      [0.5, 0.5, 0],
                      [0, 0.5, 0.5]], dtype=float)
    LP1 = dot(indice_0, basis1)
    LP1 = LP1[np.argsort(norm(LP1, axis=1))]
    LP2 = dot(R, LP1.T).T
    LP2 = LP2[np.argsort(norm(LP2, axis=1))]
    del indice
    del indice_0
    #find the lattice vectors
    RP1 = np.eye(3)
    RP1.dtype = 'float'
    RP2 = np.eye(3)
    RP2.dtype = 'float'
    
    #1.get the primitive vector along gb normal
    RP1[:,0] = gb
    RP2[:,0] = gb
    count = 0
    Found = False
    
    #2.get the minimum primitive vectors in the gb plane        
    while not Found:
        if abs(dot(LP1[count], gb)) < tol:
            RP1[:,1] = LP1[count]
            Found = True 
        count += 1
    if not Found:
        print('failed to find primitive vectors in the gb plane')
        sys.exit()
    
    count = 0    
    Found = False        
    while not Found:
        if (abs(dot(LP1[count], gb)) < tol) and (1 - abs(ang(LP1[count],RP1[:,1])) > 0.00001) :
            RP1[:,2] = LP1[count]
            Found = True 
        count += 1  
        
    if not Found:
        print('failed to find primitive vectors in the gb plane')
        sys.exit()
                
    count = 0        
    Found = False        
    while not Found:
        if abs(dot(LP2[count], gb)) < tol:
            RP2[:,1] = LP2[count]
            Found = True
        count += 1
    if not Found:
        print('failed to find primitive vectors in the gb plane')
        sys.exit()
            
    count = 0    
    Found = False        
    while not Found:
        if (abs(dot(LP2[count], gb)) < tol) and (1 - abs(ang(LP2[count],RP2[:,1])) > 0.00001) :
            RP2[:,2] = LP2[count]
            Found = True
        count += 1
        
    if not Found:
        print('failed to find primitive vectors in the gb plane')
        sys.exit()
                    
    #confirm the two lattices are right handed
    if ang(RP1[:,2], cross(RP1[:,1], RP1[:,0])) < 0:
        RP1[:,2] = - RP1[:,2]

    if ang(RP2[:,2], cross(RP2[:,1], RP2[:,0])) < 0:
        RP2[:,2] = - RP2[:,2]

    #Calculate the reciprocal Lattices
    R1 = np.eye(3)
    R2 = np.eye(3)
    M = RP1
    V = dot(M[:,0], cross(M[:,1],M[:,2]))
    R1[:,0] = cross(M[:,1],M[:,2])/V
    R1[:,1] = cross(M[:,2],M[:,0])/V
    R1[:,2] = cross(M[:,0],M[:,1])/V 
    
    M = RP2
    V = dot(M[:,0], cross(M[:,1],M[:,2]))
    R2[:,0] = cross(M[:,1],M[:,2])/V
    R2[:,1] = cross(M[:,2],M[:,0])/V
    R2[:,2] = cross(M[:,0],M[:,1])/V     
  
    #get the CSL of the 2D reciprocal lattice
    basis1 = R1.T
    basis2 = np.vstack((R2[:,1], R2[:,2]))
    
    #meshes
    lim = 1000
    x = np.arange(-lim, lim + 1, 1)
    y = x
    
    indice = (np.stack(np.meshgrid(x, y)).T).reshape(len(x) ** 2, 2)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)] 
    indice_0 = np.array(indice_0,dtype = np.float64)
    #Reciprocal lattice points in the GB plane
    LP2 = dot(indice_0, basis2)
    
    #describe lattice-2 in coordinate-1
    L2f = dot(LP2, inv(basis1))
    
    #get CSL points
    L2f = L2f[np.where(np.all(abs(np.round(L2f) - L2f) < 1e-7, axis = 1))[0]]
    
    #convert to cartesian coordinate
    L2f = dot(L2f, basis1)
    L2f = L2f[np.argsort(norm(L2f, axis=1))]
    
    v1 = np.array([1,1,1])
    v2 = np.array([1,1,1])
    
    #get the two minimum non linear vectors
    v1 = L2f[0]
    
    Found = False
    for i in L2f:
        if (1 - abs(ang(i, v1)) > tol):
            v2 = i
            Found = True
            break
            
    if not Found:
        print('failed to find the csl of the reciprocal lattices in the gb plane')
        sys.exit()    
    
    M[:,1] = v1
    M[:,2] = v2
    
    #convert to direct lattice
    cnid = np.eye(3)
    V = dot(M[:,0], cross(M[:,1],M[:,2]))
    cnid[:,0] = cross(M[:,1],M[:,2])/V
    cnid[:,1] = cross(M[:,2],M[:,0])/V
    cnid[:,2] = cross(M[:,0],M[:,1])/V        
    
    return cnid
        
def searchbyR(R, basis):
    """
    --- by Yaoshu ---
    find primitive CSL by rotation matrix R
    arguments:
    R -- rotation matrix of CSL
    basis -- lattice basis
    """
    M = np.zeros([3,3])
    basis = Basis('fcc')
    basis = np.delete(basis, 0, axis = 0)
    #meshes
    lim = 100
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    tol = 0.0001
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]    
    
    Lattice1 = dot(indice_0, basis)
    Lattice2 = np.around(dot(R, Lattice1.T).T, 8)        
    #convert Lattice2 into fcc basis and see whether the coefficients are integer
    L2f = dot(Lattice2.copy(), inv(basis))
    index = np.where(np.all(abs(np.round(L2f) - L2f) < 1e-7, axis = 1))[0]

    nn = Lattice2[index]
    TestVecs = nn[np.argsort(norm(nn, axis=1))]
    Found = False
    count = 0
    while (not Found) and count < len(TestVecs) - 1:
        M[:,2] = TestVecs[count]
        count += 1
        for i in range(len(TestVecs)):
            if 1 - ang(TestVecs[i], M[:,2]) > tol:
                M[:,1] = TestVecs[i]
                for j in range(len(TestVecs)):
                    if (1 - ang(TestVecs[j], M[:,2]) > tol) and (1 - ang(TestVecs[j], M[:,1]) > tol):
                        if (ang(TestVecs[j], cross(M[:,2], M[:,1])) > tol):
                            M[:,0] = TestVecs[j]

                            #confirm right handed
                            if ang(cross(M[:,0], M[:,1]), M[:,2]) < 0:
                                M[:,2] = - M[:,2]
                            Found = True
                            break
                if Found:
                    break
       
    
    if Found:
        return M
    else:
        return None

def Basis(basis):
    """
    --- by harddian ---
    defines the basis.
    """
    # Cubic basis
    if str(basis) == 'fcc':
        basis = np.array([[0, 0, 0],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0],
                          [0, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'bcc':
        basis = np.array([[0, 0, 0],
                          [0.5, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'sc':
        basis = np.eye(3)

    elif str(basis) == 'diamond':
        basis = np.array([[0, 0, 0],
                          [0.5, 0, 0.5],
                          [0.5, 0.5, 0],
                          [0, 0.5, 0.5],
                          [0.25, 0.25, 0.25],
                          [0.75, 0.25, 0.75],
                          [0.75, 0.75, 0.25],
                          [0.25, 0.75, 0.75]], dtype=float)
    else:
        print('Sorry! For now only works for cubic lattices ...')
        sys.exit()

    return basis

       
def Find_Monoclinic_CellbyR(R, basis, GB1):
    """
    --- by Yaoshu ---
    finds Monoclinic Cells from the primitive CSL cell.
    arguments:
    R -- rotation matrix of CSL
    basis -- lattice basis
    GB1 -- GB plane normal
    """
    tol = 0.00000001
    # Changeable limit
    lim = 150
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    MonoCell_1 = np.zeros([3,3])
    MonoCell_2 = np.zeros([3,3])
    Min_1 = searchbyR(R, basis)
    CLP = dot(indice_0,Min_1.T)
    
    CLP = CLP[np.argsort(norm(CLP, axis=1))]
    
    #1.Find the minimum lattice vector along GB normal
    #print('new GB norm')
    normcross = norm(cross(CLP, GB1.T),axis=1)
    index = np.where(normcross < tol)[0]

    if len(index) < 2:
        print('cant find the vector along GB normal')
        return None, None, None
    
    axisv = CLP[index]
    GB1 = axisv[0]
    GB2 = dot(inv(R),GB1)    
    MonoCell_1[:,0] = GB1
    MonoCell_2[:,0] = GB2
    
    #find two CSL vectors in the GB plane
    normmatrix = abs(dot(CLP, GB1.T))
    index = np.where(normmatrix < tol)[0]
    planev = CLP[index]
    planev = planev[np.argsort(norm(planev, axis=1))]
    del CLP
    
    Found = False
    count = 0
    while (not Found) and (count < len(planev)):
        MonoCell_1[:,1] = planev[count]
        for i in planev:
            #favors two vectors nearly orthogonal
            if abs(ang(i,planev[count])) < 0.3 + 0.01:
                MonoCell_1[:,2] = i
                Found = True
                break
        count +=1
    
    if not Found:
        print('failed to find two appropriate CSLs in the GB plane')
        return None, None, None
        
    else:
        #confirm that the cell is right handed
        if dot(MonoCell_1[:,0], cross(MonoCell_1[:,1], MonoCell_1[:,2])) < 0:
            MonoCell_1[:,0] = -MonoCell_1[:,0]
            
        MonoCell_2 = np.round(dot(inv(R), MonoCell_1),1)
        Num = det(MonoCell_1)*4
        if abs(abs(det(MonoCell_1)) - abs(det(MonoCell_2))) < 0.000001:
            return MonoCell_1, MonoCell_2, Num

def Find_ortho_CellbyR(R, basis, GB1):
    """
    --- by Yaoshu ---
    finds orthogonal Cells from the primitive CSL cell.
    arguments:
    R -- rotation matrix of CSL
    basis -- lattice basis
    GB1 -- GB plane normal
    """
    tol = 0.00000001
    # Changeable limit
    lim = 150
    x = np.arange(-lim, lim + 1, 1)
    y = x
    z = x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    MonoCell_1 = np.zeros([3,3])
    MonoCell_2 = np.zeros([3,3])
    Min_1 = searchbyR(R, basis)
    CLP = dot(indice_0,Min_1.T)
    
    CLP = CLP[np.argsort(norm(CLP, axis=1))]
    
    #1.Find the minimum lattice vector along GB normal
    #print('new GB norm')
    normcross = norm(cross(CLP, GB1.T),axis=1)
    index = np.where(normcross < tol)[0]

    if len(index) < 2:
        print('cant find the vector along GB normal')
        return None, None, None
    
    axisv = CLP[index]
    GB1 = axisv[0]
    GB2 = dot(inv(R),GB1)    
    MonoCell_1[:,0] = GB1
    MonoCell_2[:,0] = GB2
    
    #find two CSL vectors in the GB plane
    normmatrix = abs(dot(CLP, GB1.T))
    index = np.where(normmatrix < tol)[0]
    planev = CLP[index]
    planev = planev[np.argsort(norm(planev, axis=1))]
    del CLP
    
    Found = False
    count = 0
    while (not Found) and (count < len(planev)):
        MonoCell_1[:,1] = planev[count]
        for i in planev:
            #favors two vectors nearly orthogonal
            if abs(ang(i,planev[count])) < 0.00000001:
                MonoCell_1[:,2] = i
                Found = True
                break
        count +=1
    
    if not Found:
        print('failed to find two appropriate CSLs in the GB plane')
        return None, None, None
        
    else:
        #confirm that the cell is right handed
        if dot(MonoCell_1[:,0], cross(MonoCell_1[:,1], MonoCell_1[:,2])) < 0:
            MonoCell_1[:,0] = -MonoCell_1[:,0]
            
        MonoCell_2 = np.round(dot(inv(R), MonoCell_1),1)
        Num = det(MonoCell_1)*4
        if abs(abs(det(MonoCell_1)) - abs(det(MonoCell_2))) < 0.000001:
            return MonoCell_1, MonoCell_2, Num




