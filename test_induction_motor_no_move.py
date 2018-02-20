# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:51:20 2018

@author: haiau
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:58:32 2018

@author: haiau
"""

import math
import numpy as np
import pylab as pl
import AxisymmetricElement as AE
import FEMElement as FE
import QuadElement as QE
import PolarElement as PE
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewmarkAlgorithm as NM
import Solver as sv
import IntegrationData as idat
import SingularIntegration as SI
import MeshGenerator as mg

class LinearMagneticMaterial(mat.Material):
    def __init__(self, mur, epsr, sigma, idx):
        self.mu0 = mur*4.0e-7*np.pi
        self.sigma = sigma
        self.eps = epsr*8.854187817e-12
        self.dM = np.zeros(2)
        self.idx = idx
        self.Mu = np.zeros(2)
        self.hysteresis = False
        
    def getID(self):
        return self.idx
        
class PolarMagnetic(QE.PolarQuadElement):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder,material, intData):
        QE.PolarQuadElement.__init__(self,Nodes,pd,basisFunction,\
        nodeOrder,material,intData)
        self.store = True
        self.linear = True
    
    def getB(self):
        B = np.array([self.gradu_[1,0]/self.x_[0],-self.gradu_[0,0]])
        return B
        
    def updateMat(self, material):
        self.material = material
    
    def calculateKLinear(self, K, inod, jnod, t):
        """
        Calculate Stiffness matrix K
        """

        r = self.x_[0]
        K[0,0] = self.dN_[0,inod]*self.dN_[0,jnod]
        K[0,0] += self.dN_[1,inod]*self.dN_[1,jnod]/(r*r)
        K[0,0] /= self.material.mu0
        K[0,0] *= self.getFactor()
    
    def calculateDLinear(self, D, inod, jnod, t):
        """
        Calculate Damping matrix D
        """
        D[0,0] = self.N_[inod]*self.N_[jnod]
        D *= self.material.sigma*self.getFactor()
    
    def calculateMLinear(self, M, inod, jnod, t):
        """
        Calculate Mass matrix M
        """
        M[0,0] = self.N_[inod]*self.N_[jnod]
        M *= self.material.eps*self.getFactor()
    
#    def calculateR(self, R, inod, t):
#        """
#        Calculate load matrix R
#        """
#        r = self.x_[0]
#        re = 0.0
#        if self.material.hysteresis:
#            re += self.material.Mu[0]*self.dN_[1,inod]
#            re -= self.material.Mu[1]*(self.N_[inod]/r+self.dN_[0,inod])
#        R[0] += re
        
    def calculateRe(self, R, inod, t):
        re = self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R[0] = re
        
class TriangularPolarMagnetic(PE.PolarT6Element):
    def __init__(self, Nodes, material, dtype='float64', commonData=None,nG=4):
        PE.PolarT6Element.__init__(self,Nodes,material,dtype,commonData,nG)
        self.store = True
        self.linear = True
    
    def getB(self):
        B = np.array([self.gradu_[1,0]/self.x_[0],-self.gradu_[0,0]])
        return B
        
    def updateMat(self, material):
        self.material = material
    
    def calculateKLinear(self, K, inod, jnod, t):
        """
        Calculate Stiffness matrix K
        """

        r = self.x_[0]
        K[0,0] = self.dN_[0,inod]*self.dN_[0,jnod]
        K[0,0] += self.dN_[1,inod]*self.dN_[1,jnod]/(r*r)
        K[0,0] /= self.material.mu0
        K[0,0] *= self.getFactor()
    
    def calculateDLinear(self, D, inod, jnod, t):
        """
        Calculate Damping matrix D
        """
        D[0,0] = self.N_[inod]*self.N_[jnod]
        D[0,0] *= self.material.sigma*self.getFactor()
    
    def calculateMLinear(self, M, inod, jnod, t):
        """
        Calculate Mass matrix M
        """
        M[0,0] = self.N_[inod]*self.N_[jnod]
        M[0,0] *= self.material.eps*self.getFactor()
    
#    def calculateR(self, R, inod, t):
#        """
#        Calculate load matrix R
#        """
#        r = self.x_[0]
#        re = 0.0
#        if self.material.hysteresis:
#            re += self.material.Mu[0]*self.dN_[1,inod]
#            re -= self.material.Mu[1]*(self.N_[inod]/r+self.dN_[0,inod])
#        R[0] += re
        
    def calculateRe(self, R, inod, t):
        re = self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R[0] = re        
        
Ndof = 1             
tOrder = 1
Ng = [3,3]
totalTime = 1.0e-1
nrad = 39.79351
#numberTimeSteps = nrad*10
numberTimeSteps = 200
rho_inf = 0.8
tol = 1.0e-8
load = 3.1e6*math.sqrt(2)
#load = 2045.175*math.sqrt(2)/(np.pi/8*(5.2**2-3.2**2))


def d_to_r(deg):
    return deg*2.0*np.pi/360.0

def quad_mesh_block(phi0,phi1,r0,r1,nphi,nr,bnd=0):
    if phi0 > phi1:
        phi1 += 2*np.pi
    if r0 > r1:
        r0,r1 = r1,r0
    if r0 < 0.0 or r1 < 0.0:
        raise ValueError
    if phi0 == phi1 or r0 == r1 or nphi <= 0 or nr <= 0:
        raise ValueError
    phia = np.linspace(phi0,phi1,nphi*2+1)
    ra = np.linspace(r0,r1,nr*2+1)
    R,PHI = np.meshgrid(ra,phia)
    elements = []
    nodes = []
    for iphi in range(nphi*2+1):
        for ir in range(nr*2+1):
            nodes.append([R[iphi,ir],PE.reduce_2pi(PHI[iphi,ir])])
    numr = nr*2+1
    for iphi in range(nphi):
        for ir in range(nr):
            it = []
            st = iphi*2*numr+ir*2
            it.append(st)
            it.append(st+1)
            it.append(st+2)
            it.append(st+numr)
            it.append(st+numr+1)
            it.append(st+numr+2)
            it.append(st+numr*2)
            it.append(st+numr*2+1)
            it.append(st+numr*2+2)
            elements.append(it)
    if bnd == 1:
        belm = []
        # boundary on inner face
        for iphi in range(nphi):
            it = []
            st = iphi*2*numr
            it.append(st)
            it.append(st+numr)
            it.append(st+numr*2)
            belm.append(it)
        return nodes,elements,belm
    elif bnd == 2:
        belm = []
        # boundary on outer face
        for iphi in range(nphi):
            it = []
            st = iphi*2*numr+numr-1
            it.append(st)
            it.append(st+numr)
            it.append(st+numr*2)
            belm.append(it)
        return nodes,elements,belm
    return nodes,elements,None

def quad_mesh_ring(r0,r1,n,nr,bnd=0):
    if r0 > r1:
        r0,r1 = r1,r0
    if r0 == r1 or n <= 1 or nr <= 0:
        raise ValueError
    nodes,elements,belm = quad_mesh_block(0.0,2.0*np.pi,r0,r1,n,nr,bnd)
    nlastnodes = nr*2+1
    lastnode = len(nodes)
    lastnodes = list(range(lastnode-nlastnodes,lastnode))
    nodes = nodes[0:-nlastnodes]
    for e in elements:
        for i in range(len(e)):
            try:
                idx = lastnodes.index(e[i])
                e[i] = idx
            except ValueError:
                pass
            
    if bnd != 0:
        for e in belm:
            for i in range(len(e)):
                try:
                    idx = lastnodes.index(e[i])
                    e[i] = idx
                except ValueError:
                    pass
        return nodes,elements,belm
    return nodes,elements,None
    
def t6_mesh_pizza(r,n):
    if n <= 1 or r <= 0.0:
        raise ValueError
    phia = np.linspace(0.0,2.0*np.pi,n*2+1)
    phia = phia[0:-1]
    phib = np.linspace(0.0,2.0*np.pi,n+1)
    phib = phib[0:-1]
    nodes = [[0.0,0.0]]
    elements = []
    for phi in phia:
        nodes.append([r,phi])
    for phi in phib:
        nodes.append([r*0.5,phi])
    tolnod = n*2+1
    for i in range(n):
        it = [0]
        it.append(i*2+1)
        it.append((i*2+2)%(tolnod-1)+1)
        it.append(tolnod+(i+1)%n)
        it.append(tolnod+i)
        it.append(i*2+2)
        elements.append(it)
    return nodes,elements
    
def polarMesh(r0, r1, phi0, phi1 , nphi, nr,\
Ndof = 1,timeOrder=0,id_number=0,dtype='float64',typ='quad',bnd=0):
    if typ == 'pizzat6':
        ns,els = t6_mesh_pizza(r1,nphi)
    elif typ == 'quad':
        ns,els,belm = quad_mesh_block(phi0,phi1,r0,r1,nphi,nr,bnd)
    elif typ == 'quadring':
        ns,els,belm = quad_mesh_ring(r0,r1,nphi,nr,bnd)
    else:
        raise ValueError
    nodes = []
    for n in ns:
        nodes.append(PE.PolarNode(n,Ndof,timeOrder,id_number,dtype))
    elements = []
    for e in els:
        elements.append([])
        for ie in e:
            elements[-1].append(nodes[ie])
            
    if bnd != 0:
        belms = []
        for e in belm:
            belms.append([])
            for ie in e:
                belms[-1].append(nodes[ie])
        return elements,belms
            
    return elements,None

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)

def loadfuncA(x,t):
    return load*math.sin(60.0*2*np.pi*t)
#    return load
def loadfuncAm(x,t):
    return -load*math.sin(60.0*2*np.pi*t)
#    return -load

def loadfuncB(x,t):
    return load*math.sin(60.0*2*np.pi*t-2.0*np.pi/3.0)
#    return 0.0
    
def loadfuncBm(x,t):
    return -load*math.sin(60.0*2*np.pi*t-2.0*np.pi/3.0)
#    return 0.0
    
def loadfuncC(x,t):
    return load*math.sin(60.0*2*np.pi*t-4.0*np.pi/3.0)
#    return 0.0
    
def loadfuncCm(x,t):
    return -load*math.sin(60.0*2*np.pi*t-4.0*np.pi/3.0)
#    return 0.0

loadfunc = [loadfuncA,loadfuncBm,loadfuncC,loadfuncAm,loadfuncB,loadfuncCm]

def create_mesh():
    r0 = 0.0
    r1 = 2.0e-2
    r2 = 3.0e-2
    r3 = 3.2e-2
    r4 = 5.2e-2
    r5 = 5.7e-2
    r6 = 1.0e-1
    phi = np.array([22.5,37.5,82.5,97.5,142.5,157.5,202.5,217.5,262.5,277.5,322.5,337.5])
    phi = d_to_r(phi)
    mesh = FM.Mesh()
    basisf = QE.LagrangeBasis1D
    
    for i in range(-1,len(phi)-1):
        if i%2 == 1:
            nr = 4
        else:
            nr = 2
        elements,belm = polarMesh(r0,r1,phi[i],phi[i+1],nr,2,Ndof,tOrder,typ='quad',bnd=1)
        mat_rotor = LinearMagneticMaterial(30.0,0.0,3.72e6,0)
        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_rotor,intDat) for e in elements]
        mesh.addElements(elm)
        
        elements,belm = polarMesh(r1,r2,phi[i],phi[i+1],nr,2,Ndof,tOrder,typ='quad',bnd=1)
        mat_ind = LinearMagneticMaterial(1.0,0.0,3.72e7,1)
        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_ind,intDat) for e in elements]
        mesh.addElements(elm)
        
        elements,belm = polarMesh(r2,r3,phi[i],phi[i+1],nr,4,Ndof,tOrder,typ='quad',bnd=1)
        mat_air = LinearMagneticMaterial(1.0,0.0,0.0,4)
        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_air,intDat) for e in elements]
        mesh.addElements(elm)
        
        elements,belm = polarMesh(r3,r4,phi[i],phi[i+1],nr,2,Ndof,tOrder,typ='quad',bnd=1)
        mat_phase = LinearMagneticMaterial(1.0,0.0,0.0,2)
        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_phase,intDat) for e in elements]
        if i%2 == 1:
            for e in elm:
                e.setBodyLoad(loadfunc[math.ceil(i/2)],math.ceil(i/2))
#                for n in e.Nodes:
#                    n.setLoad(loadfunc[math.ceil(i/2)],0)
        mesh.addElements(elm)
#        if belm is not None:
#            belms = [PE.PolarElementGapBoundary(e,2,basisf,1.0/(np.pi*4.0e-7),normv=[-1.0,0.0]) for e in belm]
#            for e in belms:
#                e.current = False
#            mesh.addGapElements(belms)
        
        elements,belm = polarMesh(r4,r5,phi[i],phi[i+1],nr,1,Ndof,tOrder,typ='quad')
        mat_stator = LinearMagneticMaterial(30.0,0.0,0.0,3)
        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_stator,intDat) for e in elements]
        mesh.addElements(elm)
        
#        elements,belm = polarMesh(r5,r6,phi[i],phi[i+1],nr,1,Ndof,tOrder,typ='quad')
#        mat_outside = LinearMagneticMaterial(1.0,0.0,0.0,4)
#        elm = [PolarMagnetic(e,[2,2],basisf,nodeOrder,mat_outside,intDat) for e in elements]
#        mesh.addElements(elm)
    
    for n in mesh.Nodes:
        if n.getX()[0] <= 1.0e-14:
            n.setConstraint(False,0.0,0)
        if math.fabs(n.getX()[0] - r5) < 1.0e-14:
            n.setConstraint(False,0.0,0)
    
    mesh.generateID()
    return mesh

#nodes,elements = quad_mesh_block(d_to_r(337.5),d_to_r(22.5),1.0,2.0,2,2)    
#nodes,elements = quad_mesh_ring(1.0,2.0,4,2)
#nodes,elements = t6_mesh_pizza(1.0,4)

mesh = create_mesh() 
#mesh.plot()
 
output = FO.StandardFileOutput('/home/haiau/Documents/result_.dat')
#alg = NM.NonlinearAlphaAlgorithm(mesh,tOrder,output,sv.numpySolver(),\
#totalTime, numberTimeSteps,rho_inf,tol=1.0e-8)
alg = NM.LinearAlphaAlgorithm(mesh,tOrder,output,sv.numpySolver(),\
totalTime, numberTimeSteps,rho_inf)

alg.calculate()

#testout1,tout1 = output1.readOutput('/home/haiau/Documents/result_.dat',list(range(0,numberTimeSteps,4)),135,'u')
#testout1 = [t[0][0] for t in testout1]
    
def intFunc(element):
    return element.v_[0]*element.getFactor()

def intLineFunc(element):
    mu0 = element.material.mu0
    res = -1.0/mu0*element.gradu_[1]*element.gradu_[0]*element.x_[0]
    return res
    
def intTorque(element):
    mu0 = element.material.mu0
    res = -1.0/mu0*element.gradu_[1]*element.gradu_[0]*element.getFactor()
    return res

windingA1 = []
windingA2 = []   
for e in mesh:
    if e.bodyLoad == loadfuncA:
        windingA1.append(e)
    if e.bodyLoad == loadfuncAm:
        windingA2.append(e)

lineint = []
side = []
for e in mesh:
    if e.material.getID() == 4 and e.Nodes[0].getX()[0]>3.0e-2 + 1.0e-8\
    and e.Nodes[2].getX()[0]<3.2e-2 - 1.0e-8:
        lineint.append(e)
        if e.Nodes[0].getX()[0] > 3.0+1.0e8:
            side.append(4)
        else:
            side.append(2)
        

def calculateIntegrals():
    torque = []
    voltage = []        
    times = []
#    intDatL2 = idat.GaussianQuadratureOnEdge(Ng,idat.Gaussian1D,2)
#    intDatL4 = idat.GaussianQuadratureOnEdge(Ng,idat.Gaussian1D,4)
    for i in list(range(numberTimeSteps)):
        times.append(i*totalTime/numberTimeSteps)
        FO.StandardFileOutput.updateMesh('/home/haiau/Documents/result_.dat',mesh,i)
        res1 = 0.0
        for e in windingA1:
            res1 = e.integrate(intFunc,res1)
        res2 = 0.0
        for e in windingA2:
            res2 = e.integrate(intFunc,res2)
        voltage.append((res1-res2)*3.1e6/2045.175/math.sqrt(2))
        
        res3 = 0.0
        for i,e in enumerate(lineint):
#            if side[i] == 2:
#                res3 = e.integrate(intLineFunc,res3,intDatL2,edg=True)
#            else:
#                res3 = e.integrate(intLineFunc,res3,intDatL4,edg=True)
            res3 = e.integrate(intTorque,res3)
#        torque.append(res3[0]/2)
        torque.append(res3[0]/(0.1e-2))
        
    return times,voltage,torque

output.updateToMesh(mesh,30)
testnodes = [n for n in mesh.Nodes if n.getX()[0]==3.2e-2]        
testu = [n.getU()[0] for n in testnodes]
testphi = [n.getX()[1] for n in testnodes]    
testphi = testphi[5:] + testphi[0:5]    
testu = testu[5:] + testu[0:5]


#output.updateToMesh(mesh,30)
#X,Y,Z = mesh.meshgridValue([0.0,5.7e-2,0.0,2.0*np.pi-0.005],0.005,1.0e-8)
#X,Y,Z = mesh.meshgridValue([0.0,3.0,-2.0,3.0],0.02,1.0e-8)
  