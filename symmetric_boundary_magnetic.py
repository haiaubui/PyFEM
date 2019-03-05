#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:06:38 2018

@author: haiau
"""
import math
import numpy as np
import sympy as syp
import scipy.special as scpsp
import pylab as pl
import AxisymmetricElement as AE
import QuadElement as QE
import FEMElement as FE
import FEMBoundary as FB
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewtonRaphson as NR
import Solver as sv
import IntegrationData as idat
import MeshGenerator as mg
import injectionArray as ia
import SingularIntegration as SI


class LinearMagneticMaterial(mat.Material):
    def __init__(self, mur, epsr, sigma, idx):
        self.mu0 = mur*4.0e-7*np.pi
        self.sigma = sigma
        self.eps = epsr*8.854187817e-12
        self.dM = np.zeros(2)
        self.idx = idx
        self.Mu = np.zeros(2)
        self.hysteresis = False
        self.mu00 = 4.0e-7*np.pi
        
    def getID(self):
        return self.idx
        
class AxiSymMagneticBoundaryLinearX(AE.AxisymmetricStaticBoundary,FB.StraightBoundary1D):
    def calculateKLinear(self, K, i, j, t):
        K[0,1] = self.N_[i]*self.N_[j]
        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
        K[1,0] = -self.N_[i]*self.N_[j]*0.5
        K[1,0] *= self.getFactor()    
    
class AxiSymMagneticBoundaryLinear(AE.AxisymmetricStaticBoundary,FB.StraightBoundary1D):
    def calculateGreen(self, x, xp):
        if np.allclose(x,xp,rtol=1.0e-13) or math.fabs(xp[0])<1.0e-14 or\
        math.fabs(x[0])<1.0e-14:
#            self.G = 0.0
#            try:
#                self.gradG.fill(0.0)
#                self.grgrG.fill(0.0)
#            except AttributeError:
#                pass
            raise FB.SingularPoint
        r = x[0]
        rp = xp[0]
        z = x[1]
        zp = xp[1]
#        self.G = rp*self.Gfunc(r,z,rp,zp)
        self.G = self.Gfunc(r,z,rp,zp)
        if np.isnan(self.G):
            print('nan here')
#        self.gradG[0] = rp*self.Gdr(rp,zp,r,z)
#        self.gradG[1] = rp*self.Gdz(rp,zp,r,z)
        self.gradG[0] = self.Gdr(rp,zp,r,z)
        self.gradG[1] = self.Gdz(rp,zp,r,z)
#        self.gradG0[0] = self.Gdr0(rp,zp,r,z)
#        self.gradG0[1] = self.Gdz0(rp,zp,r,z)
#        self.grgrG[0,0] = self.Gdrr(rp,zp,r,z)
#        self.grgrG[0,1] = self.Gdrz(rp,zp,r,z)
#        self.grgrG[1,0] = self.Gdzr(rp,zp,r,z)
#        self.grgrG[1,1] = self.Gdzz(rp,zp,r,z)
#        self.gr0grG[0,0] = self.Gdr0r(rp,zp,r,z)
#        self.gr0grG[1,1] = self.Gdz0z(rp,zp,r,z)
#        self.gr0grG[0,1] = self.Gdz0r(rp,zp,r,z)        
#        self.gr0grG[1,0] = self.Gdr0z(rp,zp,r,z)
        if np.isnan(self.gr0grG).any():
            print('nan here')
            
    def postCalculateF(self, N_, dN_, factor, res):
#        idofA = 0
#        idofJ = 1
#        r = self.x_[0]
##        k1 = self.u_[idofA]*\
##        (self.normv[0]*self.gradG[0]+self.normv[1]*self.gradG[1])
#        k1 = self.u_[idofA]*np.dot(self.normv,self.gradG)
#        k1 *= factor*r
##        k1 += self.u_[idofA]*self.G*self.normv[0]*factor/r
#        k2 = self.u_[idofJ]*self.G
##        k2 += self.u_[idofA]*self.G*self.normv[0]/r
#        k2 *= factor*r
#        res += (k1 + k2)      
        idofA = 0
        idofJ = 1
        r = self.x_[0]
        k1 = self.u_[idofA]*\
        (self.normv[0]*self.gradG[0]+self.normv[1]*self.gradG[1])
        k1 *= factor*r
        k1 += self.u_[idofA]*self.G*self.normv[0]*factor/r
        k2 = self.u_[idofJ]*self.G
        k2 *= factor*r
        res += (k1 + k2)
        
    def subCalculateKLinear(self, K, element, i, j):
        #if np.allclose(element.xx_[0],0.0,rtol = 1.0e-14):
        #    K.fill(0.0)
        #    return
        idofA = 0
        idofJ = 1
        r0 = self.x_[0]
        r = element.xx_[0]
        wfac = self.getFactor()*r0
        wfact = element.getFactorX(element.detJ)
        wfacx = element.getFactorXR(element.detJ) 
#        K[idofA,idofA]=-self.N_[i]*element.Nx_[j]
#        K[idofA,idofA]*=\
#        np.einsum('i,ij,j',self.normv,self.gr0grG,element.normv)
#        K[idofA,idofA]*= wfac*wfacx*r
#        K[idofA,idofA] = 0.0
#        
#        K[idofA,idofJ]=self.N_[i]*element.Nx_[j]
#        K[idofA,idofJ]*=np.dot(self.normv,self.gradG0)
#        K[idofA,idofJ]*= wfac*wfacx*r
#        K[idofA,idofJ] = 0.0
        
        K[idofJ,idofA]=-self.N_[i]*element.Nx_[j]
        K[idofJ,idofA]*=np.dot(element.normv,self.gradG)
        
        K[idofJ,idofA] += -self.N_[i]*element.Nx_[j]*self.G*element.normv[0]/r
        K[idofJ,idofA]*= wfac*wfacx*r
#        K[idofA,idofJ]=K[idofJ,idofA]
#        K[idofJ,idofA]=0.0
        
        K[idofJ,idofJ]=-self.N_[i]*element.Nx_[j]*self.G
        K[idofJ,idofJ]*= wfac*wfact*r
#        K[idofJ,idofJ] = 0.0
        
        K /= self.material.mu00
        K *= 2.0*np.pi
    
    def calculateKLinear(self, K, i, j, t):
        K[0,1] = self.N_[i]*self.N_[j]
        K[0,1] *= self.getFactor()*self.x_[0]*2.0*np.pi/self.material.mu00
        K[1,0] = self.N_[i]*self.N_[j]*0.5
        K[1,0] *= self.getFactor()*self.x_[0]*2.0*np.pi
        K[1,0] /= self.material.mu00
        
#        K[0,1] = -0.5*self.N_[i]*self.N_[j]
#        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]
##        K[0,1] = 0.0 
#        K[1,0] = -0.5*self.N_[i]*self.N_[j]
#        K[1,0] *= self.getFactor()*2.0*np.pi*self.x_[0]
#        K[0,0] = -self.N_[i]*self.N_[j]*self.normv[0]
#        K[0,0] *= self.getFactor()*2.0*np.pi
#        K[1,1] = 0.0
##        K[0,0] = 0.0
#        
#        K /= self.material.mu00

class AxiSymMagnetic(AE.AxisymmetricQuadElement, QE.Quad9Element):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder,material, intData):
        AE.AxisymmetricQuadElement.__init__(self,Nodes,pd,basisFunction,\
        nodeOrder,material,intData)
        self.linear = True
    
    def getB(self):
        B = np.array([-self.gradu_[1,0],self.gradu_[0,0]+self.u_[0]/self.x_[0]])
        return B
        
    def updateMat(self, material):
        self.material = material
    
    def calculateKLinear(self, K, inod, jnod, t):
        """
        Calculate Stiffness matrix K
        """

        r = self.x_[0]
        K[0,0] = self.dN_[0,inod]*self.dN_[0,jnod]
        K[0,0] += self.N_[inod]*self.dN_[0,jnod]/r
        K[0,0] += self.dN_[0,inod]*self.N_[jnod]/r
        K[0,0] += self.N_[inod]*self.N_[jnod]/(r*r)
        K[0,0] += self.dN_[1,inod]*self.dN_[1,jnod]
        K[0,0] /= self.material.mu0
        K[0,0] *= self.getFactor()
        
#    def calculateK(self, K, inod, jnod, t):
#        r = self.x_[0]
#        # magnetization
#        if self.material.hysteresis:
#            dNs = self.dN_
#            Ns = self.N_
#            dm1 = self.material.dM[0]
#            dm2 = self.material.dM[1]
#            ke = dm2*dNs[0][inod]*dNs[0][jnod];
#            ke += dm2*Ns[inod]*dNs[0][jnod]/r;
#            ke += dm2*dNs[0][inod]*Ns[jnod]/r;
#            ke += dm2*Ns[inod]*Ns[jnod]/(r*r);
#            ke += dm1*dNs[1][inod]*dNs[1][jnod];
#            ke *= self.getFactor()
#            K[0,0] -= ke
#    
#    def calculateDLinear(self, D, inod, jnod, t):
#        """
#        Calculate Damping matrix D
#        """
#        D[0,0] = self.N_[inod]*self.N_[jnod]
#        D *= self.material.sigma*self.getFactor()
#    
#    def calculateMLinear(self, M, inod, jnod, t):
#        """
#        Calculate Mass matrix M
#        """
#        M[0,0] = self.N_[inod]*self.N_[jnod]
#        M *= self.material.eps*self.getFactor()
    
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
#        R[0] = re
        R.fill(0.0)
        R += re
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None,\
             deformed = False, deformed_factor=1.0):
        if fig is None:
            fig = pl.figure()
        
        X1 = self.Nodes[0].getX()
        X2 = self.Nodes[2].getX()
        pl.plot(np.array([X1[0],X2[0]]),np.array([X1[1],X2[1]]),col)
        
        X1 = self.Nodes[2].getX()
        X2 = self.Nodes[8].getX()
        pl.plot(np.array([X1[0],X2[0]]),np.array([X1[1],X2[1]]),col)
        
        X1 = self.Nodes[8].getX()
        X2 = self.Nodes[6].getX()
        pl.plot(np.array([X1[0],X2[0]]),np.array([X1[1],X2[1]]),col)
        
        X1 = self.Nodes[6].getX()
        X2 = self.Nodes[0].getX()
        pl.plot(np.array([X1[0],X2[0]]),np.array([X1[1],X2[1]]),col)
        
        nodes = self.Nodes
        for n in nodes:
            pl.plot(n.getX()[0],n.getX()[1],'.b')
        
        if number is not None:
            c = 0.5*(nodes[2].getX()+nodes[6].getX())
            pl.text(c[0],c[1],str(number))
        
        return fig, [nodes[0],nodes[2],nodes[8],nodes[6]]


Ndof = 2             
tOrder = 0
Ng = [3,3]
numberSteps = 1
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(6, 1, idat.Gaussian1D)
intSingDat = SI.SingularGaussian1D(24, intDatB.xg,\
SI.Gaussian_1D_Pn_Log, SI.Gaussian_1D_Pn_Log_Rat)
#intSingDat = idat.GaussianQuadrature(12, 1, idat.Gaussian1D)

condt = np.array([np.cos(np.pi/2.0+1.0e-2*np.pi),\
                  np.sin(np.pi/2.0+1.0e-2*np.pi)])
#condt = np.array([1.0e5,0.0])

def loadfunc(x, t):
    return 1.0

def loadfuncx(x, t):
    return np.array([1000.0/0.01/0.01,0.0])
nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_simple_mesh():
    nodes = []
    nodes.append(FN.Node([0.0,0.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([1.0,0.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([2.0,0.25],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([2.5,0.25],Ndof,timeOrder = tOrder))
    
    edge1 = mg.Edge(nodes[0],nodes[1])
    poly1 = edge1.extendToQuad(np.array([0.0,1.0]),1.0)
    edge2 = mg.Edge(nodes[2],nodes[3])
    poly2 = edge2.extendToQuad(np.array([0.0,1.0]),0.5)
    poly2.setBodyLoad(1.0)
    
    poly1.setDivisionEdge24(2)
    
    geo = mg.Geometry()
    geo.addPolygon(poly1)
    geo.addPolygon(poly2)
    
    mat1 = LinearMagneticMaterial(1.0,1.0,0.0,1)
    mat2 = LinearMagneticMaterial(1.0,1.0,0.0,2)
    poly1.setMaterial(mat1)
    poly2.setMaterial(mat2)
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
    elements = []
    
    load = 355.0
    
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
        if bdls[i] is not None:
            def loadfunc(x,t):
                #return load*math.sin(8.1e3*2*np.pi*t)
                return load
        else:
            loadfunc = None
        elements[i].setBodyLoad(loadfunc)
        
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    r0 = syp.sympify('r0')
    x0 = syp.sympify('x0')
    rx = syp.sympify('rx')
    xx = syp.sympify('xx')
    
    mt = (rx+r0)**2+(xx-x0)**2
    m = 4.0*rx*r0
    m = m/mt
    mtx = syp.sqrt(mt)*syp.pi
    kint = syp.elliptic_k(m)
    eint = syp.elliptic_e(m)
#    Gf = r0*((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gf = ((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gfr = syp.diff(Gf,rx)
    Gfz = syp.diff(Gf,xx)
    Gfr0 = syp.diff(Gf,r0)
    Gfz0 = syp.diff(Gf,x0)
    Gfr0r = syp.diff(Gfr,r0)
    Gfr0z = syp.diff(Gfr,x0)
    Gfz0r = syp.diff(Gfz,r0)
    Gfz0z = syp.diff(Gfz,x0)
    Gfrr = syp.diff(Gfr,rx)
    Gfrz = syp.diff(Gfr,xx)
    Gfzr = syp.diff(Gfz,rx)
    Gfzz = syp.diff(Gfz,xx)
    Gfunc = syp.lambdify((rx,xx,r0,x0),Gf,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr = syp.lambdify((rx,xx,r0,x0),Gfr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz = syp.lambdify((rx,xx,r0,x0),Gfz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrr = syp.lambdify((rx,xx,r0,x0),Gfrr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrz = syp.lambdify((rx,xx,r0,x0),Gfrz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzr = syp.lambdify((rx,xx,r0,x0),Gfzr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzz = syp.lambdify((rx,xx,r0,x0),Gfzz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0 = syp.lambdify((rx,xx,r0,x0),Gfr0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0 = syp.lambdify((rx,xx,r0,x0),Gfz0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0r = syp.lambdify((rx,xx,r0,x0),Gfr0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0z = syp.lambdify((rx,xx,r0,x0),Gfr0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0r = syp.lambdify((rx,xx,r0,x0),Gfz0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0z = syp.lambdify((rx,xx,r0,x0),Gfz0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    
    elementBs = []
    for i,e in enumerate(elems1):
        if e[1].getX()[0] == 0.0:
            continue
        elementBs.append(AxiSymMagneticBoundaryLinear(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(mat1)
        
        elementBs[-1].Gfunc = Gfunc
        elementBs[-1].Gdr = gradGr
        elementBs[-1].Gdz = gradGz
        elementBs[-1].Gdrr = gradGrr
        elementBs[-1].Gdrz = gradGrz
        elementBs[-1].Gdzr = gradGzr
        elementBs[-1].Gdzz = gradGzz
        elementBs[-1].Gdr0 = gradGr0
        elementBs[-1].Gdz0 = gradGz0
        elementBs[-1].Gdr0r = gradGr0r
        elementBs[-1].Gdr0z = gradGr0z
        elementBs[-1].Gdz0r = gradGz0r
        elementBs[-1].Gdz0z = gradGz0z
        elementBs[-1].linear = True
        
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 1)
        
    #mesh.addElements(elementBs)
    mesh.addBoundaryElements(elementBs)
    
    ndup = []
    for n in mesh.getNodes():
        xn = n.getX()
#        n1 = np.fabs(xn[0])<1.0e-14 and np.fabs(xn[1])<1.0e-14
        n2 = np.fabs(xn[0]-1.0)<1.0e-14 and np.fabs(xn[1])<1.0e-14
        n3 = np.fabs(xn[0]-1.0)<1.0e-14 and np.fabs(xn[1]-1.0)<1.0e-14
#        n4 = np.fabs(xn[0])<1.0e-14 and np.fabs(xn[1]-1.0)<1.0e-14
        n5 = np.fabs(xn[0]-2.0)<1.0e-14 and np.fabs(xn[1]-0.25)<1.0e-14
        n6 = np.fabs(xn[0]-2.5)<1.0e-14 and np.fabs(xn[1]-0.25)<1.0e-14
        n7 = np.fabs(xn[0]-2.5)<1.0e-14 and np.fabs(xn[1]-0.75)<1.0e-14
        n8 = np.fabs(xn[0]-2.0)<1.0e-14 and np.fabs(xn[1]-0.75)<1.0e-14
        if n2 or n3 or n5 or n6 or n7 or n8:
            be1 = None
            be2 = None
            for be in mesh.BoundaryElements:
                if n in be:
                    if be1 is None:
                        be1 = be
                    elif be2 is None:
                        be2 = be
                        break
            nx1 = n.copy()
            nx1.friendOF(n,0)
            nx2 = n.copy()
            nx2.friendOF(n,0)
            n.freedom[1] = False
            for i1, nt1 in enumerate(be1.Nodes):
                if n == nt1:
                    be1.Nodes[i1] = nx1
                    ndup.append(nx1)
            for i2, nt2 in enumerate(be2.Nodes):
                if n == nt2:
                    be2.Nodes[i2] = nx2
                    ndup.append(nx2)
    for n in ndup:
        mesh.Nodes.append(n)
        mesh.Nnod += 1
    
    mesh.generateID()

    
    #mesh.Nodes[4].setLoad(loadfunc,0)
    
    return mesh

def create_test_mesh():
    nodes = []
    nodes.append(FN.Node([0.0,-1.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([1.0,-1.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([2.0,-1.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([2.5,-1.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([3.0,-1.0],Ndof,timeOrder = tOrder))
    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [1.0,0.25,0.55,0.25,1.0]

    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
    
    polys = geo.getPolygons()
    
    mat1 = LinearMagneticMaterial(1.0,1.0,0.0,1)
#    mat2 = LinearMagneticMaterial(1.0,1.0,5.0e6,2)
    for p in polys:
        p.setMaterial(mat1)
    polys[12].setBodyLoad(1.0)
    
    for p in polys:
        p.setDivisionEdge13(2)
        p.setDivisionEdge24(2)
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
    load = 355.0
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0])>1.0-1.0e-13 and \
#        n.getX()[1]>0.25-1.0e-13 and \
#        n.getX()[1]<0.75+1.0e-13 and n.getX()[0]<2.5+1.0e-13:
#            n.setLoad(load,0)
#        if math.fabs(n.getX()[0]-2.25)<1.0e-13 and\
#        math.fabs(n.getX()[1]-0.525)<1.0e-13:
#            n.setLoad(load*2.0*np.pi*n.getX()[0],0)
    elements = []
    
    
    
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
        if bdls[i] is not None:
            def loadfunc(x,t):
                #return load*math.sin(8.1e3*2*np.pi*t)
                return load
        else:
            loadfunc = None
        elements[i].setBodyLoad(loadfunc)
        
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    r0 = syp.sympify('r0')
    x0 = syp.sympify('x0')
    rx = syp.sympify('rx')
    xx = syp.sympify('xx')
    
    mt = (rx+r0)**2+(xx-x0)**2
    m = 4.0*rx*r0
    m = m/mt
    mtx = syp.sqrt(mt)*syp.pi
    kint = syp.elliptic_k(m)
    eint = syp.elliptic_e(m)
#    Gf = r0*((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gf = ((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gfr = syp.diff(Gf,rx)
    Gfz = syp.diff(Gf,xx)
    Gfr0 = syp.diff(Gf,r0)
    Gfz0 = syp.diff(Gf,x0)
    Gfr0r = syp.diff(Gfr0,rx)
    Gfr0z = syp.diff(Gfr0,xx)
    Gfz0r = syp.diff(Gfz0,rx)
    Gfz0z = syp.diff(Gfz0,xx)
    Gfrr = syp.diff(Gfr,rx)
    Gfrz = syp.diff(Gfr,xx)
    Gfzr = syp.diff(Gfz,rx)
    Gfzz = syp.diff(Gfz,xx)
    Gfunc = syp.lambdify((rx,xx,r0,x0),Gf,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr = syp.lambdify((rx,xx,r0,x0),Gfr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz = syp.lambdify((rx,xx,r0,x0),Gfz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrr = syp.lambdify((rx,xx,r0,x0),Gfrr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrz = syp.lambdify((rx,xx,r0,x0),Gfrz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzr = syp.lambdify((rx,xx,r0,x0),Gfzr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzz = syp.lambdify((rx,xx,r0,x0),Gfzz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0 = syp.lambdify((rx,xx,r0,x0),Gfr0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0 = syp.lambdify((rx,xx,r0,x0),Gfz0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0r = syp.lambdify((rx,xx,r0,x0),Gfr0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0z = syp.lambdify((rx,xx,r0,x0),Gfr0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0r = syp.lambdify((rx,xx,r0,x0),Gfz0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0z = syp.lambdify((rx,xx,r0,x0),Gfz0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    
    elementBs = []
    for i,e in enumerate(elems1):
        if e[1].getX()[0] == 0.0:
            continue
        elementBs.append(AxiSymMagneticBoundaryLinear(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(mat1)
        
        elementBs[-1].Gfunc = Gfunc
        elementBs[-1].Gdr = gradGr
        elementBs[-1].Gdz = gradGz
        elementBs[-1].Gdrr = gradGrr
        elementBs[-1].Gdrz = gradGrz
        elementBs[-1].Gdzr = gradGzr
        elementBs[-1].Gdzz = gradGzz
        elementBs[-1].Gdr0 = gradGr0
        elementBs[-1].Gdz0 = gradGz0
        elementBs[-1].Gdr0r = gradGr0r
        elementBs[-1].Gdr0z = gradGr0z
        elementBs[-1].Gdz0r = gradGz0r
        elementBs[-1].Gdz0z = gradGz0z
        elementBs[-1].linear = True
        
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 1)
        
    #mesh.addElements(elementBs)
    mesh.addBoundaryElements(elementBs)
    mesh.generateID()
    
    return mesh


def create_mesh_x():
    H = 0.001
    R = 0.1
    nodes = []
    nodes.append(FN.Node([0,-2*H*10],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R*2,-2*H*10],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R*2+0.1*R,-2*H*10],Ndof,timeOrder = tOrder))    

    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = 10*H
    
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
        
#    nodes1 = []
#    nodes1.append(FN.Node([0,H*21],Ndof,timeOrder = tOrder))
#    nodes1.append(FN.Node([R*2,H*21],Ndof,timeOrder = tOrder))
#    nodes1.append(FN.Node([R*2+0.1*R,H*21],Ndof,timeOrder = tOrder))
#    
#    edges1 = [mg.Edge(nodes1[i],nodes1[i+1]) for i in range(len(nodes1)-1)]
#    
#    for e in edges1:
#        geo.addPolygons(e.extendToQuad(d,s))
        
    nodes2 = []
    nodes2.append(FN.Node([0,0],Ndof,timeOrder = tOrder))
    nodes2.append(FN.Node([R,0],Ndof,timeOrder = tOrder))
    
    edges2 = [mg.Edge(nodes2[i],nodes2[i+1]) for i in range(len(nodes2)-1)]
    
    s = H
    
    for e in edges2:
        geo.addPolygons(e.extendToQuad(d,s))
        
    polys = geo.getPolygons()
    
    polys[0].setDivisionEdge13(20)
    polys[0].setDivisionEdge24(1)
    polys[2].setDivisionEdge13(20)
    polys[2].setDivisionEdge24(1)
    
#    polys[4].setDivisionEdge13(8)
#    polys[4].setDivisionEdge24(1)
    
    mat1 = LinearMagneticMaterial(1000.0, 0.0, 0.0,0)
    mat2 = LinearMagneticMaterial(1.0, 0.0, 1.0,1)
    mat3 = LinearMagneticMaterial(100.0, 0.0, 1.0,2)
    mat4 = LinearMagneticMaterial(1.0, 0.0, 1.0,3)
    
    polys[0].setMaterial(mat3)
    polys[1].setMaterial(mat2)
    polys[2].setMaterial(mat1)
#    polys[3].setMaterial(mat2)
#    polys[4].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:            
        if math.fabs(n.getX()[0])<1.0e-14:
#            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
            
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
#        elements.append(AxiMechElement(e,[2,2],QE.LagrangeBasis1D,\
#        nodeOrder,m,intDat,condt))
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat))
        if m.idx == 1:
            elements[-1].setBodyLoad(loadfuncx)
#        if m.idx == 0:
#            elements[-1].setBodyLoad(loadfuncG)
#        elements[-1].setBodyLoad(loadfunc)
    
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec, bndMat] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof,get_mat=True)
    
    r0 = syp.sympify('r0')
    x0 = syp.sympify('x0')
    rx = syp.sympify('rx')
    xx = syp.sympify('xx')
    
    mt = (rx+r0)**2+(xx-x0)**2
    m = 4.0*rx*r0
    m = m/mt
    mtx = syp.sqrt(mt)*syp.pi
    kint = syp.elliptic_k(m)
    eint = syp.elliptic_e(m)
#    Gf = r0*((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gf = ((2.0-m)*kint-2.0*eint)/(m*mtx)
    Gfr = syp.diff(Gf,rx)
    Gfz = syp.diff(Gf,xx)
    Gfr0 = syp.diff(Gf,r0)
    Gfz0 = syp.diff(Gf,x0)
    Gfr0r = syp.diff(Gfr,r0)
    Gfr0z = syp.diff(Gfr,x0)
    Gfz0r = syp.diff(Gfz,r0)
    Gfz0z = syp.diff(Gfz,x0)
    Gfrr = syp.diff(Gfr,rx)
    Gfrz = syp.diff(Gfr,xx)
    Gfzr = syp.diff(Gfz,rx)
    Gfzz = syp.diff(Gfz,xx)
    Gfunc = syp.lambdify((rx,xx,r0,x0),Gf,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr = syp.lambdify((rx,xx,r0,x0),Gfr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz = syp.lambdify((rx,xx,r0,x0),Gfz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrr = syp.lambdify((rx,xx,r0,x0),Gfrr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGrz = syp.lambdify((rx,xx,r0,x0),Gfrz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzr = syp.lambdify((rx,xx,r0,x0),Gfzr,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGzz = syp.lambdify((rx,xx,r0,x0),Gfzz,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0 = syp.lambdify((rx,xx,r0,x0),Gfr0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0 = syp.lambdify((rx,xx,r0,x0),Gfz0,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0r = syp.lambdify((rx,xx,r0,x0),Gfr0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGr0z = syp.lambdify((rx,xx,r0,x0),Gfr0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0r = syp.lambdify((rx,xx,r0,x0),Gfz0r,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    gradGz0z = syp.lambdify((rx,xx,r0,x0),Gfz0z,\
    modules=['numpy',{'elliptic_k':scpsp.ellipk,'elliptic_e':scpsp.ellipe}])
    
    elementBs = []
    for i,e in enumerate(elems1):
        if np.fabs(e[1].getX()[0]) < 1.0e-13:
            continue
        elementBs.append(AxiSymMagneticBoundaryLinear(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(bndMat[i])
        
#        if bndMat[i].idx == 0:
#            elementBs[-1].deformed=True
        
#        if np.fabs(e[1].getX()[1])<1.0e-14:
#            elementBs[-1].normv = np.array([0.0,-1.0])
#        if np.fabs(e[1].getX()[1]-H)<1.0e-14:
#            elementBs[-1].normv = np.array([0.0,1.0])
#        if np.fabs(e[1].getX()[0]-R)<1.0e-14:
#            elementBs[-1].normv = np.array([1.0,0.0])
        elementBs[-1].Gfunc = Gfunc
        elementBs[-1].Gdr = gradGr
        elementBs[-1].Gdz = gradGz
        elementBs[-1].Gdrr = gradGrr
        elementBs[-1].Gdrz = gradGrz
        elementBs[-1].Gdzr = gradGzr
        elementBs[-1].Gdzz = gradGzz
        elementBs[-1].Gdr0 = gradGr0
        elementBs[-1].Gdz0 = gradGz0
        elementBs[-1].Gdr0r = gradGr0r
        elementBs[-1].Gdr0z = gradGr0z
        elementBs[-1].Gdz0r = gradGz0r
        elementBs[-1].Gdz0z = gradGz0z
#        elementBs[-1].linear=False
    

    mesh.addBoundaryElements(elementBs)   
    
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 1)
    
#    ndup = []
#    for n in mesh.getNodes():
#        xn = n.getX()
#        n1 = np.fabs(xn[0]-0.2)<1.0e-14 and np.fabs(xn[1]+0.02)<1.0e-14
#        n2 = np.fabs(xn[0]-0.2)<1.0e-14 and np.fabs(xn[1]+0.01)<1.0e-14
#        n3 = np.fabs(xn[0]-0.2)<1.0e-14 and np.fabs(xn[1]-0.011)<1.0e-13
#        n4 = np.fabs(xn[0]-0.2)<1.0e-14 and np.fabs(xn[1]-0.021)<1.0e-13
#        if n1 or n2 or n3 or n4:
#            be1 = None
#            be2 = None
#            for be in mesh.BoundaryElements:
#                if n in be:
#                    if be1 is None:
#                        be1 = be
#                    elif be2 is None:
#                        be2 = be
#                        break
#            nx1 = n.copy()
#            nx1.setConstraint(False,0.0,0)
#            nx1.setConstraint(False,0.0,1)
#            nx1.friendOF(n,2)
#            nx2 = n.copy()
#            nx2.setConstraint(False,0.0,0)
#            nx2.setConstraint(False,0.0,1)
#            nx2.friendOF(n,2)
#            n.freedom[3] = False
#            for i1, nt1 in enumerate(be1.Nodes):
#                if n == nt1:
#                    be1.Nodes[i1] = nx1
#                    ndup.append(nx1)
#            for i2, nt2 in enumerate(be2.Nodes):
#                if n == nt2:
#                    be2.Nodes[i2] = nx2
#                    ndup.append(nx2)
#    for n in ndup:
#        mesh.Nodes.append(n)
#        mesh.Nnod += 1
    mesh.generateID()
    return mesh

mesh = create_mesh_x()
#mesh = create_simple_mesh()
#mesh = create_test_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result_BI_Linear.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg = NR.LinearStabilityProblem(mesh,output,sv.numpySolver())
alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
                                          arcl=1.0,max_iter=200)
#alg.enableVariableConstraint()
#alg.calculate(True)
alg.calculate(False)
#alg.calculate()

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)
        