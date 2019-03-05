#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:56:52 2018

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
import FEMAlgorithm as FA
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
        
class SymMagneticBoundaryLinear(FB.StandardStaticBoundary,FB.StraightBoundary1D):
    def calculateGreen(self, x, xp):
        if np.allclose(x,xp,rtol=1.0e-13):
#            self.G = 0.0
#            try:
#                self.gradG.fill(0.0)
#                self.grgrG.fill(0.0)
#            except AttributeError:
#                pass
            raise FB.SingularPoint
        x0 = x[0]
        xx = xp[0]
        y0 = x[1]
        yx = xp[1]        

        self.G = self.Gfunc(xx,yx,x0,y0)
        if np.isnan(self.G):
            print('nan here')
#        self.gradG[0] = rp*self.Gdr(rp,zp,r,z)
#        self.gradG[1] = rp*self.Gdz(rp,zp,r,z)
        self.gradG[0] = self.Gdx(xx,yx,x0,y0)
        self.gradG[1] = self.Gdy(xx,yx,x0,y0)
        self.gradG0[0] = self.Gdx0(xx,yx,x0,y0)
        self.gradG0[1] = self.Gdy0(xx,yx,x0,y0)
        self.grgrG[0,0] = self.Gdxx(xx,yx,x0,y0)
        self.grgrG[0,1] = self.Gdxy(xx,yx,x0,y0)
        self.grgrG[1,0] = self.Gdyx(xx,yx,x0,y0)
        self.grgrG[1,1] = self.Gdyy(xx,yx,x0,y0)
        self.gr0grG[0,0] = self.Gdx0x(xx,yx,x0,y0)
        self.gr0grG[1,1] = self.Gdy0y(xx,yx,x0,y0)
        self.gr0grG[0,1] = self.Gdx0y(xx,yx,x0,y0)        
        self.gr0grG[1,0] = self.Gdy0x(xx,yx,x0,y0)
        if np.isnan(self.gr0grG).any():
            print('nan here')
            
    def postCalculateF(self, N_, dN_, factor, res):
        idofA = 0
        idofJ = 1
        k1 = self.u_[idofA]*np.dot(self.normv,self.gradG)
        k1 *= factor
        if self.jbasis:
            k2 = self.uj_[idofJ]*self.G
        else:
            k2 = self.u_[idofJ]*self.G
        k2 *= factor
        res += (k1 - k2)        
        
    def subCalculateKLinear(self, K, element, i, j):
        #if np.allclose(element.xx_[0],0.0,rtol = 1.0e-14):
        #    K.fill(0.0)
        #    return
        idofA = 0
        idofJ = 1
        wfac = self.getFactor()
        wfact = element.getFactorX(element.detJ)
        wfacx = element.getFactorXR(element.detJ) 
        K[idofA,idofA]=-self.N_[i]*element.Nx_[j]
        K[idofA,idofA]*=\
        np.einsum('i,ij,j',self.normv,self.gr0grG,element.normv)
        K[idofA,idofA]*= wfac*wfacx
#        K[idofA,idofA]=np.dot(self.tangent,self.dN_[:,i])
#        K[idofA,idofA]*=np.dot(element.tangent,element.dNx_[:,j])*self.G
#        K[idofA,idofA]*= wfac*wfact
#        K[idofA,idofA] = 0.0
        
        K[idofA,idofJ]=self.N_[i]*element.Njx_[j]
        K[idofA,idofJ]*=np.dot(self.normv,self.gradG0)
        K[idofA,idofJ]*= wfac*wfact
#        K[idofA,idofJ] = 0.0
        
        K[idofJ,idofA]=self.Nj_[i]*element.Nx_[j]
        K[idofJ,idofA]*=np.dot(element.normv,self.gradG)
#        K[idofJ,idofA]=self.Nj_[i]*np.dot(element.tangent,element.dNx_[j])
#        K[idofJ,idofA]*=self.G
        K[idofJ,idofA]*= wfac*wfact
#        K[idofA,idofJ]=K[idofJ,idofA]
#        K[idofJ,idofA]=0.0
        
        K[idofJ,idofJ]=-self.Nj_[i]*element.Njx_[j]*self.G
        K[idofJ,idofJ]*= wfac*wfact
#        K[idofJ,idofJ] = 0.0
#        K[idofA,idofA] = -K[idofJ,idofJ]
        
        K /= self.material.mu0
    
    def calculateKLinear(self, K, i, j, t):
#        K[0,1] = self.N_[i]*self.N_[j]
#        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
#        K[1,0] = -self.N_[i]*self.N_[j]*0.5
#        K[1,0] *= self.getFactor()
#        K[1,0] /= self.material.mu0
        
        K[0,1] = -0.5*self.N_[i]*self.Nj_[j]
        K[0,1] *= self.getFactor()
#        K[0,1] = 0.0 
        K[1,0] = -0.5*self.Nj_[i]*self.N_[j]
        K[1,0] *= self.getFactor()
        K[1,1] = 0.0
        K[0,0] = 0.0
        
        K /= self.material.mu0

class Magnetic2D(QE.Quad9Element):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder,material, intData):
        QE.QuadElement.__init__(self,Nodes,pd,basisFunction,\
        nodeOrder,material,intData)
        self.linear = True
    
    def getB(self):
        B = np.array([-self.gradu_[1,0],self.gradu_[0,0]])
        return B
        
    def updateMat(self, material):
        self.material = material
    
    def calculateKLinear(self, K, inod, jnod, t):
        """
        Calculate Stiffness matrix K
        """

        K[0,0] = self.dN_[0,inod]*self.dN_[0,jnod]
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
        R[0] = re
        
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
intSingDat = SI.SingularGaussian1D(48, intDatB.xg,\
SI.Gaussian_1D_Pn_Log_Rat, SI.Gaussian_1D_Pn_Log_Rat_Rat2)
#intSingDat = idat.GaussianQuadrature(60, 1, idat.Gaussian1D)


def loadfunc(x, t):
    return 1.0
nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

#commonD = FB.BoundaryCommonData(3, 2, intDatB, jbasisF=QE.RampBasis1D3N,\
#                                intSingData = intSingDat)
#commonD = FB.BoundaryCommonData(3, 2, intDatB, jbasisF=QE.PulseBasis1D,\
#                                intSingData = intSingDat)
commonD = FB.BoundaryCommonData(3, 2, intDatB)

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
    
#    poly1.setDivisionEdge24(2)
    
    geo = mg.Geometry()
    geo.addPolygon(poly1)
    geo.addPolygon(poly2)
    
    mat1 = LinearMagneticMaterial(1.0,1.0,0.0,1)
    mat2 = LinearMagneticMaterial(1.0,1.0,0.0,2)
    poly1.setMaterial(mat1)
    poly2.setMaterial(mat2)
    
    load = 355.0
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
#    for n in nodesx:
##        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
##            n.setConstraint(False, 0.0, 0)
##            n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0]-2.25)<1.0e-13 and\
#        math.fabs(n.getX()[1]-0.5)<1.0e-13:
#            n.setLoad(load,0)
    elements = []
    
    
    
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(Magnetic2D(e,[2,2],QE.LagrangeBasis1D,\
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
    
    x0 = syp.sympify('x0')
    y0 = syp.sympify('y0')
    xx = syp.sympify('xx')
    yy = syp.sympify('yy')
    
    mt = (x0-xx)**2+(yy-y0)**2
    Gf = -syp.log(mt)/(4.0*syp.pi)
    Gfx = syp.diff(Gf,xx)
    Gfy = syp.diff(Gf,yy)
    Gfx0 = syp.diff(Gf,x0)
    Gfy0 = syp.diff(Gf,y0)
    Gfx0x = syp.diff(Gfx0,xx)
    Gfx0y = syp.diff(Gfx0,yy)
    Gfy0x = syp.diff(Gfy0,xx)
    Gfy0y = syp.diff(Gfy0,yy)
    Gfxx = syp.diff(Gfx,xx)
    Gfxy = syp.diff(Gfx,yy)
    Gfyx = syp.diff(Gfy,xx)
    Gfyy = syp.diff(Gfy,yy)
    Gfunc = syp.lambdify((xx,yy,x0,y0),Gf,modules=['numpy'])
    gradGx = syp.lambdify((xx,yy,x0,y0),Gfx,modules=['numpy'])
    gradGy = syp.lambdify((xx,yy,x0,y0),Gfy,modules=['numpy'])
    gradGxx = syp.lambdify((xx,yy,x0,y0),Gfxx,modules=['numpy'])
    gradGxy = syp.lambdify((xx,yy,x0,y0),Gfxy,modules=['numpy'])
    gradGyx = syp.lambdify((xx,yy,x0,y0),Gfyx,modules=['numpy'])
    gradGyy = syp.lambdify((xx,yy,x0,y0),Gfyy,modules=['numpy'])
    gradGx0 = syp.lambdify((xx,yy,x0,y0),Gfx0,modules=['numpy'])
    gradGy0 = syp.lambdify((xx,yy,x0,y0),Gfy0,modules=['numpy'])
    gradGx0x = syp.lambdify((xx,yy,x0,y0),Gfx0x,modules=['numpy'])
    gradGx0y = syp.lambdify((xx,yy,x0,y0),Gfx0y,modules=['numpy'])
    gradGy0x = syp.lambdify((xx,yy,x0,y0),Gfy0x,modules=['numpy'])
    gradGy0y = syp.lambdify((xx,yy,x0,y0),Gfy0y,modules=['numpy'])
    
    elementBs = []
    for i,e in enumerate(elems1):
#        if e[1].getX()[0] == 0.0:
#            continue
        elementBs.append(SymMagneticBoundaryLinear(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i,\
        commonData = commonD))
        elementBs[-1].setMaterial(mat1)
        
        elementBs[-1].Gfunc = Gfunc
        elementBs[-1].Gdx = gradGx
        elementBs[-1].Gdy = gradGy
        elementBs[-1].Gdxx = gradGxx
        elementBs[-1].Gdxy = gradGxy
        elementBs[-1].Gdyx = gradGyx
        elementBs[-1].Gdyy = gradGyy
        elementBs[-1].Gdx0 = gradGx0
        elementBs[-1].Gdy0 = gradGy0
        elementBs[-1].Gdx0x = gradGx0x
        elementBs[-1].Gdx0y = gradGx0y
        elementBs[-1].Gdy0x = gradGy0x
        elementBs[-1].Gdy0y = gradGy0y
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
        n1 = np.fabs(xn[0])<1.0e-14 and np.fabs(xn[1])<1.0e-14
        n2 = np.fabs(xn[0]-1.0)<1.0e-14 and np.fabs(xn[1])<1.0e-14
        n3 = np.fabs(xn[0]-1.0)<1.0e-14 and np.fabs(xn[1]-1.0)<1.0e-14
        n4 = np.fabs(xn[0])<1.0e-14 and np.fabs(xn[1]-1.0)<1.0e-14
        n5 = np.fabs(xn[0]-2.0)<1.0e-14 and np.fabs(xn[1]-0.25)<1.0e-14
        n6 = np.fabs(xn[0]-2.5)<1.0e-14 and np.fabs(xn[1]-0.25)<1.0e-14
        n7 = np.fabs(xn[0]-2.5)<1.0e-14 and np.fabs(xn[1]-0.75)<1.0e-14
        n8 = np.fabs(xn[0]-2.0)<1.0e-14 and np.fabs(xn[1]-0.75)<1.0e-14
        if n1 or n2 or n3 or n4 or n5 or n6 or n7 or n8:
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
#        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
#            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0])>1.0-1.0e-13 and \
#        n.getX()[1]>0.25-1.0e-13 and \
#        n.getX()[1]<0.75+1.0e-13 and n.getX()[0]<2.5+1.0e-13:
#            n.setLoad(load,0)
        if math.fabs(n.getX()[0]-2.25)<1.0e-13 and\
        math.fabs(n.getX()[1]-0.525)<1.0e-13:
            n.setLoad(load,0)
    elements = []
    
    
    
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(Magnetic2D(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
#        if bdls[i] is not None:
#            def loadfunc(x,t):
#                #return load*math.sin(8.1e3*2*np.pi*t)
#                return load
#        else:
#            loadfunc = None
#        elements[i].setBodyLoad(loadfunc)
        
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    x0 = syp.sympify('x0')
    y0 = syp.sympify('y0')
    xx = syp.sympify('xx')
    yy = syp.sympify('yy')
    
    mt = (x0-xx)**2+(yy-y0)**2
    Gf = -syp.log(mt)/(4.0*syp.pi)
    Gfx = syp.diff(Gf,xx)
    Gfy = syp.diff(Gf,yy)
    Gfx0 = syp.diff(Gf,x0)
    Gfy0 = syp.diff(Gf,y0)
    Gfx0x = syp.diff(Gfx0,xx)
    Gfx0y = syp.diff(Gfx0,yy)
    Gfy0x = syp.diff(Gfy0,xx)
    Gfy0y = syp.diff(Gfy0,yy)
    Gfxx = syp.diff(Gfx,xx)
    Gfxy = syp.diff(Gfx,yy)
    Gfyx = syp.diff(Gfy,xx)
    Gfyy = syp.diff(Gfy,yy)
    Gfunc = syp.lambdify((xx,yy,x0,y0),Gf,modules=['numpy'])
    gradGx = syp.lambdify((xx,yy,x0,y0),Gfx,modules=['numpy'])
    gradGy = syp.lambdify((xx,yy,x0,y0),Gfy,modules=['numpy'])
    gradGxx = syp.lambdify((xx,yy,x0,y0),Gfxx,modules=['numpy'])
    gradGxy = syp.lambdify((xx,yy,x0,y0),Gfxy,modules=['numpy'])
    gradGyx = syp.lambdify((xx,yy,x0,y0),Gfyx,modules=['numpy'])
    gradGyy = syp.lambdify((xx,yy,x0,y0),Gfyy,modules=['numpy'])
    gradGx0 = syp.lambdify((xx,yy,x0,y0),Gfx0,modules=['numpy'])
    gradGy0 = syp.lambdify((xx,yy,x0,y0),Gfy0,modules=['numpy'])
    gradGx0x = syp.lambdify((xx,yy,x0,y0),Gfx0x,modules=['numpy'])
    gradGx0y = syp.lambdify((xx,yy,x0,y0),Gfx0y,modules=['numpy'])
    gradGy0x = syp.lambdify((xx,yy,x0,y0),Gfy0x,modules=['numpy'])
    gradGy0y = syp.lambdify((xx,yy,x0,y0),Gfy0y,modules=['numpy'])
    
    elementBs = []
    for i,e in enumerate(elems1):
#        if e[1].getX()[0] == 0.0:
#            continue
        elementBs.append(SymMagneticBoundaryLinear(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i,\
        commonData = commonD))
        elementBs[-1].setMaterial(mat1)
        
        elementBs[-1].Gfunc = Gfunc
        elementBs[-1].Gdx = gradGx
        elementBs[-1].Gdy = gradGy
        elementBs[-1].Gdxx = gradGxx
        elementBs[-1].Gdxy = gradGxy
        elementBs[-1].Gdyx = gradGyx
        elementBs[-1].Gdyy = gradGyy
        elementBs[-1].Gdx0 = gradGx0
        elementBs[-1].Gdy0 = gradGy0
        elementBs[-1].Gdx0x = gradGx0x
        elementBs[-1].Gdx0y = gradGx0y
        elementBs[-1].Gdy0x = gradGy0x
        elementBs[-1].Gdy0y = gradGy0y
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


mesh = create_simple_mesh()
#mesh = create_test_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg = FA.LinearStaticAlgorithm(mesh,output,sv.numpySolver())
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
        