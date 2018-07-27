#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:39:07 2018

@author: haiau
"""

import math
import numpy as np
import pylab as pl
import AxisymmetricElement as AE
import QuadElement as QE
import FEMElement as FE
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewtonRaphson as NR
import Solver as sv
import IntegrationData as idat
import MeshGenerator as mg
import injectionArray as ia


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
    def __init__(self, Emod, nu, rho, mur,idx):
        self.Emod = Emod
        self.nu = nu
        self.rho = rho
        self.idx = idx
        self.lmd = Emod*nu/(1+nu)/(1-2*nu)
        self.mu = Emod/2.0/(1+nu)
        bigI = np.zeros((3,3,3,3),np.float64)
        smallI = np.zeros((3,3,3,3),np.float64)
        for i in range(0,3,1):
            for j in range(0,3,1):
                for k in range(0,3,1):
                    for n in range(0,3,1):
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
#        smallI *= 2*self.mu*self.lmd/(2*self.mu+self.lmd)
        self.Cmat = bigI + smallI
        self.mu0 = mur*4.0e-7*np.pi
        self.mur = mur
        self.mu00 = 4.0e-7*np.pi
        self.Cmat_flat = np.zeros((4,4),np.float64)
        self.Cmat_flat[0,0] = 2*self.mu+self.lmd
        self.Cmat_flat[1,1] = 2*self.mu+self.lmd
        self.Cmat_flat[2,2] = 2*self.mu+self.lmd
        self.Cmat_flat[3,3] = self.mu
        self.Cmat_flat[0,1] = self.lmd
        self.Cmat_flat[1,0] = self.lmd
        self.Cmat_flat[0,2] = self.lmd
        self.Cmat_flat[2,0] = self.lmd
        self.Cmat_flat[1,2] = self.lmd
        self.Cmat_flat[2,1] = self.lmd
        
    def getID(self):
        return self.idx
    
class AxiMechElement(AE.AxisymmetricQuadElement,QE.Quad9Element):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, material, intData,\
                 condt = [0.0,0.0]):
        AE.AxisymmetricQuadElement.__init__(self,Nodes,pd,basisFunction,\
                                            nodeOrder,material,intData)
        self.rotNs = []
        for i in range(self.intData.getNumberPoint()):
            self.rotNs.append(np.empty((self.Ndim,self.Nnod),self.dtype))
        
#        self.RiLambd = []
#        for i in range(self.Nnod):
#            self.RiLambd.append(ia.zeros(self.Ndof,self.dtype))
        self.rotA = np.empty(self.Ndim,self.dtype)
        self.Fg = np.empty((3,3),self.dtype)
        self.Fmg = np.empty((3,3),self.dtype)
        self.Eg = np.empty((3,3),self.dtype)
        self.Sg = np.empty((3,3),self.dtype)
        self.Fgx = np.empty((2,2),dtype=self.dtype)
        self.Fmgx = np.empty((2,2),dtype=self.dtype)
        self.Cgx = np.empty((2,2),dtype=self.dtype)
        self.ident = np.eye(3,dtype = self.dtype)
        self.ident1= np.zeros((2,3),dtype = self.dtype)
        self.ident2= np.zeros((2,3),dtype = self.dtype)
        self.ident1[0,0] = 1.0
        self.ident2[0,1] = 1.0
        self.ident1[1,2] = 1.0
        self.graduM_ = np.zeros((3,3),dtype = self.dtype)
        self.temp_graduM_ = np.zeros((3,3),dtype = self.dtype)
        self.E_eng = np.zeros(4,dtype=self.dtype)
        self.E_eng_temp_ = np.zeros(4,dtype=self.dtype)
        
#        self.KtT = []
#        self.KtS = []
#        for i in range(self.Nnod):
#            self.KtT.append([])
#            self.KtS.append([])
#            for j in range(self.Nnod):
#                self.KtT[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
#                self.KtS[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
        
        
        self.kt = np.empty((2,2),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(2,self.dtype)
        self.condt = condt
#        self.linear = True
    
    def connect(self, *arg):
        AE.AxisymmetricQuadElement.connect(self, *arg)
#        KtT = arg[9]
#        KtS = arg[10]
#        for inod in range(self.Nnod):
#            ids1 = self.Nodes[inod].getID()
#            for jnod in range(self.Nnod):
#                ids2 = self.Nodes[jnod].getID()
#                for idof in range(self.Ndof):
#                    id1 = ids1[idof]
#                    if id1 >= 0:
#                        for jdof in range(self.Ndof):
#                            id2 = ids2[jdof]
#                            if id2 >= 0:
#                                self.KtT[inod][jnod].connect((idof,jdof),KtT,\
#                                (id1,id2))
#                                self.KtS[inod][jnod].connect((idof,jdof),KtS,\
#                                (id1,id2))
#        RiLamb = arg[8]
#        for inod in range(self.Nnod):
#            ids1 = self.Nodes[inod].getID()
#            for idof in range(self.Ndof):
#                id1 = ids1[idof]
#                if id1 >= 0:
#                    self.RiLambd[inod].connect(idof,RiLamb,id1)
    
    def getBasis(self):
        AE.AxisymmetricQuadElement.getBasis(self)
        self.rotN = self.rotNs[self.ig]
        self.gradTens = self.gradX[self.ig]
        self.Bmat = self.BmatA[self.ig]

    def getGradUP(self, gradu, dN_ = None):
        FE.Element.getGradUP(self,gradu,dN_)
        if dN_ is None:
            dN_ = self.dN_
        self.graduM_.fill(0.0)
#        self.E_eng.fill(0.0)
#        self.E_eng[0] = self.gradu_[0,0]
#        self.E_eng[1] = self.u_[0]/self.x_[0]
#        self.E_eng[2] = self.gradu_[1,1]
#        self.E_eng[3] = self.gradu_[0,1] + self.gradu_[1,0]
        for i in range(self.Nnod):
            np.einsum('i,ijk',self.Nodes[i].getU().toNumpy()[0:2],\
                      self.gradTens[i,:,:,:],out=self.temp_graduM_)
            self.graduM_ += self.temp_graduM_
            taolao=np.dot(self.Bmat[i,:,:],self.Nodes[i].getU().toNumpy()[0:2])
            self.E_eng += taolao
            
    def getGradU(self, gradu, data, dN_ = None):
#        FE.Element.getGradU(self,gradu,data,dN_)
        self.getGradUP(gradu,dN_)
#        self.E_eng.fill(0.0)
#        self.E_eng[0] = self.gradu_[0,0]
#        self.E_eng[1] = self.u_[0]/self.x_[0]
#        self.E_eng[2] = self.gradu_[1,1]
#        self.E_eng[3] = self.gradu_[0,1] + self.gradu_[1,0]
    
    def updateMat(self, material):
        self.material = material
        
    def prepareElement(self):
        self.calculateBasis(self.nodeOrder)
        self.gradX = []
        x = np.empty(self.Ndim,self.dtype)
        for ig in range(self.intData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Ns_[ig])
            self.gradX.append(np.zeros((self.Nnod,2,3,3),dtype = self.dtype))
            for inod in range(self.Nnod):
                gradx = np.array([self.dNs_[ig][0,inod],\
                                  0.0,self.dNs_[ig][1,inod]])
                gradxx = np.einsum('ij,k',self.ident1,gradx)
                self.gradX[ig][inod,:,:,:] += gradxx
                if math.fabs(x[0]) > 1.0e-14:
                    gradx = np.array([0,self.Ns_[ig][inod]/x[0],0.0])
                    gradxx = np.einsum('ij,k',self.ident2,gradx)
                    self.gradX[ig][inod,:,:,:] += gradxx
                
        self.BmatA = []
        for ig in range(self.intData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Ns_[ig])
            self.BmatA.append(np.zeros((self.Nnod,4,2),dtype = self.dtype))
            for inod in range(self.Nnod):
                self.BmatA[ig][inod,0,0]=self.dNs_[ig][0,inod]
                if math.fabs(x[0]) > 1.0e-14:
                    self.BmatA[ig][inod,1,0]=self.Ns_[ig][inod]/x[0]
                self.BmatA[ig][inod,2,1]=self.dNs_[ig][1,inod]
                self.BmatA[ig][inod,3,0]=self.dNs_[ig][1,inod]
                self.BmatA[ig][inod,3,1]=self.dNs_[ig][0,inod]
        
        for ig in range(self.intData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Ns_[ig])
            for i in range(self.Nnod):
                self.rotNs[ig][0,i]= -self.dNs_[ig][1,i]
                if math.fabs(x[0]) < 1.0e-14:
                    self.rotNs[ig][1,i] = 0.0
                    continue
                try:
                    self.rotNs[ig][1,i]= self.Ns_[ig][i]/x[0] + self.dNs_[ig][0,i]
                except IndexError:
                    print('error here')
        
    def calculateFES(self):
        """
        calculate deformation gradient F
        """
        self.rotA[0] = -self.gradu_[1,2]
        if math.fabs(self.x_[0]) > 1.0e-14:
            self.rotA[1] = self.u_[2]/self.x_[0] + self.gradu_[0,2]
        else:
            self.rotA[1] = 0.0
        np.copyto(self.Fg,self.graduM_)
        self.Fg += self.ident
        self.Fgx[0,0] = self.Fg[0,0]
        self.Fgx[0,1] = self.Fg[0,2]
        self.Fgx[1,0] = self.Fg[2,0]
        self.Fgx[1,1] = self.Fg[2,2]
        self.JF = self.Fg[0,0]*self.Fg[2,2]-self.Fg[0,2]*self.Fg[2,0]
        self.JF *= self.Fg[1,1]
        self.Fmg.fill(0.0)
        self.Fmg[0,0] = self.Fg[2,2]*self.Fg[1,1]/self.JF
        self.Fmg[2,2] = self.Fg[0,0]*self.Fg[1,1]/self.JF
        self.Fmg[0,2] = -self.Fg[0,2]*self.Fg[1,1]/self.JF
        self.Fmg[2,0] = -self.Fg[2,0]*self.Fg[1,1]/self.JF
        self.Fmg[1,1] = self.Fg[0,0]*self.Fg[2,2]-self.Fg[2,0]*self.Fg[0,2]
        self.Fmg[1,1] /= self.JF
        self.Fmgx[0,0] = self.Fmg[0,0]
        self.Fmgx[1,1] = self.Fmg[2,2]
        self.Fmgx[0,1] = self.Fmg[0,1]
        self.Fmgx[1,0] = self.Fmg[1,0]
        self.Cg = np.dot(self.Fg.T,self.Fg)
        self.Cgx[0,0] = self.Cg[0,0]
        self.Cgx[0,1] = self.Cg[0,2]
        self.Cgx[1,0] = self.Cg[2,0]
        self.Cgx[1,1] = self.Cg[2,2]
        self.Eg = self.Cg - self.ident
        self.Eg *= 0.5
        np.einsum('ijkl,kl',self.material.Cmat,self.Eg,out=self.Sg)
    
    def multiplyCderiv(self, a,b,jnod):
        kt = np.einsum('i,kli,lj,j',a,self.gradTens[jnod,:,:,:],self.Fg,b)
        kt += np.einsum('i,li,klj,j',a,self.Fg,self.gradTens[jnod,:,:,:],b)
        kt /= self.JF
        return kt
    
    def multiplyFmderiv(self, inod, jnod):
        kt = np.einsum('kij,im,nj,lmn',self.gradTens[inod,:,:,:],self.Fmg,\
                       self.Fmg,self.gradTens[jnod,:,:,:])
        kt *= -1.0
        return kt
        
    def derivativeJinv(self, jnod):
        kt = np.einsum('mn,knm',self.Fmg,self.gradTens[jnod,:,:,:])
        kt *= -1.0/self.JF
        return kt
    
    def multiplyFderiv(self,b,c,inod,jnod):
        kt1 = np.einsum('pkq,q',self.gradTens[jnod,:,:,:],c)
        kt2 = np.einsum('jik,i',self.gradTens[inod,:,:,:],b)
        kt = np.dot(kt2,kt1)
        kt /= self.JF
        return kt
        
#    def calculateKLinear(self, K, inod, jnod, t):
#        """
#        Calculate Stiffness matrix K
#        """
#
#        kt = np.einsum('ijk,jkmn,lnm',self.gradTens[inod,:,:,:],\
#                  self.material.Cmat,self.gradTens[jnod,:,:,:])
#        kt *= self.getFactor()
#        K[0,0] = kt[0,0]
#        K[0,1] = kt[0,1]
#        K[1,0] = kt[1,0]
#        K[1,1] = kt[1,1]
        
    def calculateK(self, K, inod, jnod, t):
        #Mechanics part--------------------------------------------------------
        kg = np.einsum('jim,kin,nm',self.gradTens[inod,:,:,:],\
                       self.gradTens[jnod,:,:,:],self.Sg)
        np.einsum('ji,kjl,ilmn,qpm,pn',self.Fg,self.gradTens[inod,:,:,:],\
                  self.material.Cmat,self.gradTens[jnod,:,:,:],self.Fg,\
                  out=self.kt)
        self.kt += kg
        
        self.kt *= self.getFactor()
        K[0,0] += self.kt[0,0]
        K[0,1] += self.kt[0,1]
        K[1,0] += self.kt[1,0]
        K[1,1] += self.kt[1,1]
        
#        self.KtT[inod][jnod][0,0] += self.kt[0,0]
#        self.KtT[inod][jnod][0,1] += self.kt[0,1]
#        self.KtT[inod][jnod][1,0] += self.kt[1,0]
#        self.KtT[inod][jnod][1,1] += self.kt[1,1]
        
        #Magneto-mechanics part------------------------------------------------
        condt = np.array([self.rotA[0],0.0,self.rotA[1]])
#        condt = np.array([self.condt[0],0.0,self.condt[1]])
        
        # Derivative F^-1
#        en = np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        en = np.einsum('i,ij,j',condt,self.Cg,condt)
        en *= 0.5/self.JF
        kt1212 = self.multiplyFmderiv(inod,jnod)
        kt1212 *= en
        
        # Derivative C
        ktx = 0.5*self.multiplyCderiv(condt,condt,jnod)
#        ktx *= (2.0-self.material.mur)
        kt1212 += np.einsum('kij,ij,l',self.gradTens[inod,:,:,:],self.Fmg,ktx)

        # Derivative 1/J
#        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)
        te = en * self.Fmg
        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
        ktx = np.einsum('kij,ij',self.gradTens[inod,:,:,:],te)
        kty = self.derivativeJinv(jnod)
        kt1212 += np.einsum('i,j',ktx,kty)
        
        # Derivative F
        ktxx = self.multiplyFderiv(condt,condt,inod,jnod)
        kt1212 -= ktxx
        
        kt1212 *= self.getFactor()
        kt1212 /= self.material.mu0
        kt1212 *= (2.0-self.material.mur)
#        kt1212 *= -1.0
        
        K[0,0] -= kt1212[0,0]
        K[0,1] -= kt1212[0,1]
        K[1,0] -= kt1212[1,0]
        K[1,1] -= kt1212[1,1]
        
#        kt33 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotN[:,jnod])
        kt33 = np.einsum('i,i',self.rotN[:,inod],self.rotN[:,jnod])
#        kt33 /= self.JF*self.material.mu0
        kt33 /= self.material.mu0
        kt33 *= self.getFactor()
        
        K[2,2] += kt33
        
        rotN = np.array([self.rotN[0,inod],0.0,self.rotN[1,inod]])        
        kt321 = self.multiplyCderiv(rotN,condt,jnod)
        kt321 /= self.material.mu0*self.JF
        en = np.einsum('i,ij,j',rotN,self.Cg,condt)/self.material.mu0
        kt321 += en*self.derivativeJinv(jnod)
        kt321 *= self.getFactor()
        
        K[2,0] += kt321[0]
        K[2,1] += kt321[1]
        
        
        rotN = np.array([self.rotN[0,jnod],0.0,self.rotN[1,jnod]]) 
#        en = np.einsum('i,ij,j',rotN,self.Cg,condt)*(2.0-self.material.mur)
        en = np.einsum('i,ij,j',rotN,self.Cg,condt)
        kt123 = en*np.einsum('ijl,jl',self.gradTens[inod,:,:,:],self.Fmg)
        
        te = np.einsum('ijl,j,lk,k',self.gradTens[inod,:,:,:],rotN,self.Fg,\
                       condt)
        te += np.einsum('ijl,j,lk,k',self.gradTens[inod,:,:,:],condt,self.Fg,\
                       rotN)
        kt123 -= te
        kt123 *= self.getFactor()
        kt123 /= self.JF
        kt123 *= (2.0-self.material.mur)/self.material.mu0
#        kt123 *= -1.0
        K[0,2] -= kt123[0]
        K[1,2] -= kt123[1]
    
    
    def calculateR(self, R, inod, t):
        """
        Calculate load matrix R
        """
        #Mechanics part
        np.einsum('in,kij,nj',self.Fg,self.gradTens[inod,:,:,:],self.Sg,\
                  out=self.rie)
        self.rie *= self.getFactor()
        R[0] += self.rie[0]
        R[1] += self.rie[1]
        
        #Magneto-mechanics part
        condt = np.array([self.rotA[0],0.0,self.rotA[1]])
#        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)
        te = en*self.Fmg
        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
        te /= self.material.mu0
        te *= (2.0-self.material.mur)
        te /= self.JF
        rie = np.einsum('kij,ij',self.gradTens[inod,:,:,:],te)
        
        rie *= self.getFactor()
        
#        rie *= -1.0
                
        R[0] -= rie[0]
        R[1] -= rie[1]
        
        r3 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotA)
#        r3 = np.einsum('i,i',self.rotN[:,inod],self.rotA)
        r3 /= self.JF*self.material.mu0
#        r3 /= self.material.mu0
        R[2] += r3*self.getFactor()
        
Ndof = 3             
tOrder = 0
Ng = [3,3]
numberSteps = 10
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)

condt = 50.0*np.array([np.cos(np.pi/2.0),\
                  np.sin(np.pi/2.0)])
#condt = np.array([1.0e5,0.0])

def loadfunc(x, t):
    return 9.8

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh():
    H = 0.01
    R = 0.1
    Ho = 0.1
    Ro = 0.2
    nodes = []
    nodes.append(FN.Node([0,0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R,0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([Ro,0],Ndof,timeOrder = tOrder))
    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [Ho,H,Ho]
    
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
        
    nepl = 4    
    ney = 2
    polys = geo.getPolygons()
    polys[0].setDivisionEdge13(nepl)
    polys[0].setDivisionEdge24(ney)
    polys[1].setDivisionEdge13(nepl)
    polys[1].setDivisionEdge24(1)
    polys[2].setDivisionEdge13(nepl)
    polys[2].setDivisionEdge24(ney)
    
    polys[3].setDivisionEdge13(1)
    polys[3].setDivisionEdge24(ney)
    polys[4].setDivisionEdge13(1)
    polys[4].setDivisionEdge24(1)
    polys[5].setDivisionEdge13(1)
    polys[5].setDivisionEdge24(ney)
    
    mat1 = LinearMechanicMaterial(2.1e11, 0.3, 7.8e3,10000.0,0)
    mat2 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,1)
    for i in range(6):
        if i != 1:
            polys[i].setMaterial(mat2)
        else:
            polys[i].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:
        if n.getX()[0] > R+1.0e-8:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
        if n.getX()[1] < Ho-1.0e-8:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
        if n.getX()[1] > Ho+H+1.0e-8:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
#        n.setConstraint(False, 0.0, 0)
#        n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0]-R)<1.0e-14 and math.fabs(n.getX()[1]-(H*0.5+Ho))<1.0e-13:
#            ncenter = n
        
        if math.fabs(n.getX()[0]-R)<1.0e-14 :
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            
        if math.fabs(n.getX()[0])<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False,0.0,2)
        if math.fabs(n.getX()[1])<1.0e-14 or\
        math.fabs(n.getX()[1]-(H+2*Ho))<1.0e-13:
            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            n.controlVar(2)
        if math.fabs(n.getX()[0]-Ro)<1.0e-14 :
            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            n.controlVar(2)
            
#    for n in nodesx:
#        if math.fabs(n.getX()[0]-R)<1.0e-14 and n.getX()[1]>Ho-1.0e-8\
#        and n.getX()[1]<Ho+H+1.0e-8:
#            n.friendOF(ncenter,0)
            
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiMechElement(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat,condt))
#        elements[-1].setBodyLoad(loadfunc)
    
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    
        
    return mesh



mesh = create_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg = NR.LinearStabilityProblem(mesh,output,sv.numpySolver())
#alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
#                                          arcl=10.0,max_iter=200)
alg = NR.VariableControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg.enableVariableConstraint()
alg.calculate(True)
#alg.calculate(False)
#alg.calculate()

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)
        
