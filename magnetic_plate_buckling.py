#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:45:53 2018

@author: haiau
"""

import math
import numpy as np
import pylab as pl
import FEMBoundary as FB
import QuadElement as QE
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewtonRaphson as NR
import Solver as sv
import IntegrationData as idat
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
    
class LinearMechanicMaterial(mat.Material):
    def __init__(self, Emod, nu, rho, mur):
        self.Emod = Emod
        self.nu = nu
        self.rho = rho
        self.lmd = Emod*nu/(1+nu)/(1-2*nu)
        self.mu = Emod/2.0/(1+nu)
        bigI = np.zeros((2,2,2,2),np.float64)
        smallI = np.zeros((2,2,2,2),np.float64)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for n in range(2):
                        part1 = 0
                        if(i == n and j == k):
                            part1 = 1
                        part2 = 0
                        if(i == k and j == n):
                            part2 = 1
                        bigI[i,j,k,n] = part1 + part2
                        if(i == j and k == n):
                            smallI[i,j,k,n] = 1
                        else:
                            smallI[i,j,k,n] = 0
        bigI *= self.mu
        smallI *= self.lmd
        self.Cmat = bigI + smallI
        self.mu0 = mur*4.0e-7*np.pi
        self.mu00 = 4.0e-7*np.pi
        self.mur = mur
        
    def getID(self):
        return self.idx
    
class MechElement(QE.Quad9Element):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, \
                 material, intData, condt = np.array([0.0,0.0])):
        QE.Quad9Element.__init__(self,Nodes,pd,basisFunction,\
                                            nodeOrder,material,intData)
        self.rotNs = []
        for i in range(self.intData.getNumberPoint()):
            self.rotNs.append(np.empty((self.Ndim,self.Nnod),self.dtype))
        self.rotA = np.empty(self.Ndim,self.dtype)
        self.Fg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Fmg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Eg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Sg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.ident = np.eye(self.Ndim,dtype = self.dtype)
        self.kt = np.empty((2,2),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(2,self.dtype)
        self.condt = condt
    
    def getBasis(self):
        QE.Quad9Element.getBasis(self)
        self.rotN = self.rotNs[self.ig]
    
    def updateMat(self, material):
        self.material = material
        
    def prepareElement(self):
        self.calculateBasis(self.nodeOrder)
        x = np.empty(self.Ndim,self.dtype)
        for ig in range(self.intData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Ns_[ig])
            for i in range(self.Nnod):
                self.rotNs[ig][0,i]= -self.dNs_[ig][1,i]
                self.rotNs[ig][1,i]= self.dNs_[ig][0,i]
        
    def calculateFES(self):
        """
        calculate deformation gradient F
        """
#        self.rotA[0] = -self.gradu_[1,2]
#        self.rotA[1] = self.gradu_[0,2]
        np.copyto(self.Fg,self.gradu_[0:2,0:2].T)
        self.Fg += self.ident
        self.JF = self.Fg[0,0]*self.Fg[1,1]-self.Fg[0,1]*self.Fg[1,0]
        self.Fmg[0,0] = self.Fg[1,1]/self.JF
        self.Fmg[1,1] = self.Fg[0,0]/self.JF
        self.Fmg[0,1] = -self.Fg[0,1]/self.JF
        self.Fmg[1,0] = -self.Fg[1,0]/self.JF
        self.Cg = np.dot(self.Fg.T,self.Fg)
        self.Eg = self.Cg - self.ident
        self.Eg *= 0.5
        np.einsum('ijkl,kl',self.material.Cmat,self.Eg,out=self.Sg)
    
    def multiplyCderiv(self, a,b,jnod):
        kt = np.einsum('i,i,kj,j',a,self.dN_[:,jnod],self.Fg,b)
        kt += np.einsum('i,ki,j,j',a,self.Fg,self.dN_[:,jnod],b)
        kt /= self.JF
        return kt
    
    def multiplyFmderiv(self, inod, jnod):
        en = -np.einsum('i,ij,j',self.dN_[:,jnod],self.Fmg,self.dN_[:,inod])
        kt = en*self.Fmg
        return kt
        
    def multiplyFderiv(self,b,c,inod,jnod):
        en = np.dot(c,self.dN_[:,jnod])
        kt = en*np.einsum('i,j',b,self.dN_[:,inod])
        kt /= self.JF
        return kt
    
    def derivativeJinv(self, jnod):
        return -1.0/(self.JF*self.JF)*np.dot(self.dN_[:,jnod],self.Fmg)
#    def calculateKLinear(self, K, inod, jnod, t):
#        """
#        Calculate Stiffness matrix K
#        """
#
#        r = self.x_[0]
#        K[0,0] = self.dN_[0,inod]*self.dN_[0,jnod]
#        K[0,0] += self.N_[inod]*self.dN_[0,jnod]/r
#        K[0,0] += self.dN_[0,inod]*self.N_[jnod]/r
#        K[0,0] += self.N_[inod]*self.N_[jnod]/(r*r)
#        K[0,0] += self.dN_[1,inod]*self.dN_[1,jnod]
#        K[0,0] /= self.material.mu0
#        K[0,0] *= self.getFactor()
        
    def calculateK(self, K, inod, jnod, t):
        kg = np.einsum('i,ij,j',self.dN_[:,inod],self.Sg,self.dN_[:,jnod]);
        np.einsum('ij,k,jklm,l,nm',self.Fg,self.dN_[:,inod],\
                  self.material.Cmat,self.dN_[:,jnod],self.Fg,out=self.kt)
        self.kt += kg*self.ident
        self.kt *= self.getFactor()
        K[0,0] += self.kt[0,0]
        K[0,1] += self.kt[0,1]
        K[1,0] += self.kt[1,0]
        K[1,1] += self.kt[1,1]
        
        # Stress tensor coupling
        kt33 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cg,self.rotN[:,jnod])
        kt33 /= self.JF*self.material.mu0
#        kt33 = np.einsum('i,i',self.rotN[:,inod],self.rotN[:,jnod])
        
#        test
#        r = self.x_[0]
#        Ktest = self.dN_[0,inod]*self.dN_[0,jnod]
#        Ktest += self.N_[inod]*self.dN_[0,jnod]/r
#        Ktest += self.dN_[0,inod]*self.N_[jnod]/r
#        Ktest += self.N_[inod]*self.N_[jnod]/(r*r)
#        Ktest += self.dN_[1,inod]*self.dN_[1,jnod]
#        Ktest /= self.material.mu0
#        kt33 = Ktest
#        endtest
        
#        kt33 = np.dot(self.rotN[:,inod],self.rotN[:,jnod])
#        kt33 /= self.material.mu0
        kt33 *= self.getFactor()
        kt321 = self.multiplyCderiv(self.rotN[:,inod],self.rotA,jnod)
        kt321 /= self.material.mu0
        kt321 *= self.getFactor()
        
        condt = t*self.condt
#        condt = 1.0*self.rotA
        en = np.einsum('i,ij,j',condt,self.Cg,condt)
        en *= 0.5/(self.JF)
        kt1212 = self.multiplyFmderiv(inod,jnod)
        kt1212 *= en
        ktx = self.multiplyCderiv(condt,condt,jnod)
        kty = np.dot(self.Fmg,self.dN_[:,inod])
        kt1212 += 0.5*np.einsum('i,j',kty,ktx)
        ktxx = self.multiplyFderiv(condt,condt,inod,jnod)
        kt1212 -= ktxx
        
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)
        te = en*self.Fmg
        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
        ktj = self.derivativeJinv(jnod)
        kt1212 += np.einsum('kj,j,l',te,self.dN_[:,inod],ktj)
        
        kt1212 *= (2.0-self.material.mur)
        kt1212 *= self.getFactor()/self.material.mu0
        
        en1 = np.einsum('i,ij,j',self.rotN[:,jnod],self.Cg,self.rotA)
        en1 += np.einsum('i,ij,j',self.rotA,self.Cg,self.rotN[:,jnod])
        en1 /= 2.0*self.JF
        kt123 = np.dot(self.Fmg,self.dN_[:,inod])
        kt123 *= en1
        en2 = np.einsum('i,ij,j',self.dN_[:,inod],self.Fg,self.rotA)
        en3 = np.einsum('i,ij,j',self.dN_[:,inod],self.Fg,self.rotN[:,jnod])
        kt123 -= en2/self.JF*self.rotN[:,jnod]
        kt123 -= en3/self.JF*self.rotA
        kt123 /= self.material.mu0
        kt123 *= (2.0-self.material.mur)
        kt123 *= self.getFactor()
        
        K[0,0] -= kt1212[0,0]
        K[0,1] -= kt1212[0,1]
        K[1,0] -= kt1212[1,0]
        K[1,1] -= kt1212[1,1]
        
#        K[2,2] += kt33
#        K[2,0] += kt321[0]
#        K[2,1] += kt321[1]
        
#        K[0,2] -= kt123[0]
#        K[1,2] -= kt123[1]
    
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
    
    def calculateR(self, R, inod, t):
        """
        Calculate load matrix R
        """
        np.einsum('ij,jk,k',self.Fg,self.Sg,self.dN_[:,inod],out=self.rie[0:2])
#        np.einsum('i,ijkl,kl',self.dN_[:,inod],self.material.Cmat,\
#                  self.gradu_,out=self.rie)
        self.rie *= self.getFactor()
        R[0] += self.rie[0]
        R[1] += self.rie[1]
        
        #stress tensor coupling
        condt = self.condt
#        condt = 1.0*self.rotA
        en = np.einsum('i,ij,j',condt,self.Cg,condt)
        en *= 0.5/(self.JF*self.material.mu0)
        rie = np.dot(self.Fmg,self.dN_[:,inod])
        rie *= en
        en1 = np.einsum('i,ij,j',self.dN_[:,inod],self.Fg,condt)
        en1 /= self.JF*self.material.mu0
        rie -= en1*condt
        
        rie *= (2.0-self.material.mur)
        rie *= self.getFactor()
        
        R[0] -= t*t*rie[0]
        R[1] -= t*t*rie[1]
        self.RiLambd[inod][0] += t*rie[0]
        self.RiLambd[inod][1] += t*rie[1]
        
#        R[0] -= rie[0]
#        R[1] -= rie[1]
        
        r3 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cg,self.rotA)
        r3 /= self.JF*self.material.mu0
#        r3 = np.einsum('i,i',self.rotN[:,inod],self.rotA)
#        r3 = np.dot(self.rotN[:,inod],self.rotA)
#        r3 /= self.material.mu0
#        R[2] += r3*self.getFactor()
        
#    def calculateRe(self, R, inod, t):
#        re = self.N_[inod]*self.getBodyLoad(t)
#        re *= -self.getFactor()*self.material.rho
#        R[0] = 0.0
#        R[1] = re
#        R[2] = 10.0*self.getFactor()


class MagMech(FB.NeumannBoundary,MechElement):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, intData,\
                 normv, ide, condt):
        FB.NeumannBoundary.__init__(self,Nodes,pd,basisFunction,\
                                            nodeOrder,intData,normv,ide)
        self.rotNs = []
        for i in range(self.intData.getNumberPoint()):
            self.rotNs.append(np.empty((self.Ndim,self.Nnod),self.dtype))
        self.rotA = np.empty(self.Ndim,self.dtype)
        self.Fg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Fmg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Eg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.Sg = np.empty((self.Ndim,self.Ndim),self.dtype)
        self.ident = np.eye(self.Ndim,dtype = self.dtype)
        self.kt = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(self.Ndof,self.dtype)
        self.tangent = np.cross(np.array([self.normv[0],self.normv[1],0.0]),\
                                         np.array([0.0,0.0,1.0]))[0:2]
        self.tangent /= np.linalg.norm(self.tangent)
        self.condt = condt
     
    def multiplyFmderiv(self, b, jnod):
        kt = np.einsum('n,nj,i,ik',self.dN_[:,jnod],self.Fmg,b,self.Fmg)
        kt *= -1.0
        return kt
    
    def multiplyFderiv(self,a,b,c,jnod):
        nB = np.dot(a,b)
        nB *= np.dot(self.dN_[:,jnod],c)
        kt = nB*self.ident
        kt /= self.JF
        return kt
        
    def calculateK(self, K, inod, jnod, t):
        condt = t*self.condt
        kt1212 = self.multiplyFderiv(self.normv,condt,condt,jnod)
        kt1212 *= -self.N_[inod]
        kt = self.multiplyCderiv(condt,condt,jnod)
        kt *= 0.5
        normv = np.dot(self.normv,self.Fmg)*self.N_[inod]
        kt1212 += np.einsum('i,j',normv,kt)
        ktm = self.multiplyFmderiv(self.normv,jnod)
        ktm *= self.N_[inod]
        ktm *= np.einsum('i,ij,j',condt,self.Cg,condt)*0.5/self.JF
        kt1212 += ktm
        
        te = np.einsum('i,ij,j',condt,self.Cg,condt)*np.dot(self.normv,self.Fmg)
        te -= np.dot(self.normv,condt)*np.dot(self.Fg,condt)
        te *= self.N_[inod]
        ktx = self.derivativeJinv(jnod)        
        kt1212 += np.einsum('i,j',te,ktx)
        
        kt1212 *= self.getFactor()/self.material.mu00
        
#        kt321 = self.multiplyCderiv(self.tangent,self.tangent,jnod)
#        kt321 *= np.dot(condt,self.tangent)*self.N_[inod]
#        kt321 *= -self.getFactor()/self.material.mu00
        
        K[0,0] += kt1212[0,0]
        K[0,1] += kt1212[0,1]
        K[1,0] += kt1212[1,0]
        K[1,1] += kt1212[1,1]
#        K[2,0] += kt321[0]
#        K[2,1] += kt321[1]
        
    
    def calculateR(self, R, inod, t):
        condt = 1.0*self.condt
        en = np.einsum('i,ij,j',condt,self.Cg,condt)
        en /= 2.0*self.JF
        ri12 = en*np.dot(self.normv,self.Fmg)
        en = np.dot(self.normv,condt)/self.JF
        ri12 -= en*np.dot(condt,self.Fg)
        ri12 *= self.N_[inod]*self.getFactor()/self.material.mu00
        
#        en = np.einsum('i,ij,j',self.tangent,self.Cg,self.tangent)/self.JF
#        en *= np.dot(self.tangent,condt)
#        en = np.dot(self.tangent,self.condt)
#        ri3 = -en*self.N_[inod]*self.getFactor()/self.material.mu00
        
        R[0] += t*t*ri12[0]
        R[1] += t*t*ri12[1]
        self.RiLambd[inod][0] += t*ri12[0]
        self.RiLambd[inod][1] += t*ri12[1]
#        R[2] += ri3
        
Ndof = 2             
tOrder = 0
Ng = [3,3]
numberSteps = 20
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)

condt = np.array([0.0,1.0])

def loadfunc(x, t):
    return 9.8

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh():
    H = 0.001
    R = 0.1
    nodes = []
    nodes.append(FN.Node([0,0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R,0],Ndof,timeOrder = tOrder))
    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = H
    
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
        
    polys = geo.getPolygons()
    polys[0].setDivisionEdge13(4)
    polys[0].setDivisionEdge24(1)
    
    mat1 = LinearMechanicMaterial(2.1e11, 0.3, 7.8e3,10000.0)
    
    polys[0].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:
#        n.setConstraint(False, 0.0, 0)
#        n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0]-R)<1.0e-14:
#            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False, 0.0, 1)
#            n.setConstraint(False, condt[1]*n.getX()[0],2)
        if math.fabs(n.getX()[0])<1.0e-14 and math.fabs(n.getX()[1]-H*0.5)<1.0e-14:
            n.setConstraint(False, 0.0, 1)
        if math.fabs(n.getX()[0])<1.0e-14:
            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False,0.0,2)
#        if math.fabs(n.getX()[1])<1.0e-14 or\
#        math.fabs(n.getX()[1]-H)<1.0e-14:
#            n.setConstraint(False, condt[1]*n.getX()[0],2)
            
            
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(MechElement(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat,condt))
    
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    elementBs = []
    for i,e in enumerate(elems1):
        elementBs.append(MagMech(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,normBndVec[i],i,condt))
        elementBs[-1].setMaterial(mat1)
    

    mesh.addBoundaryElements(elementBs)    
        
    return mesh

def contraction21412(A,b,C,d,E,out=None):
    if out is None:
        out = np.zeros([2,2],np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    for m in range(2):
                        for n in range(2):
                            out[i,j] += A[i,k]*b[l]*C[k,l,m,n]*d[m]*E[j,n]
    return out

mesh = create_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
                                          arcl=100.0)
#alg.enableVariableConstraint()
alg.calculate(True)

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)
        