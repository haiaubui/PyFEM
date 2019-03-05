#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:02:22 2018

@author: haiau
"""

import math
import numpy as np
import sympy as syp
import scipy.special as scpsp
import pylab as pl
import Element.AxisymmetricElement as AE
import Element.QuadElement as QE
import Element.FEMElement as FE
import Element.FEMBoundary as FB
import Mesh.FEMNode as FN
import Mesh.FEMMesh as FM
import InOut.FEMOutput as FO
import Material.Material as mat
import Algorithm.NewtonRaphson as NR
import Math.Solver as sv
import Math.IntegrationData as idat
import Mesh.MeshGenerator as mg
import Math.injectionArray as ia
import Math.SingularIntegration as SI


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
        try:
            for i in range(self.Nnod):
                np.einsum('i,ijk',self.Nodes[i].getU().toNumpy()[0:2],\
                          self.gradTens[i,:,:,:],out=self.temp_graduM_)
                self.graduM_ += self.temp_graduM_
        except AttributeError:
            pass
#            gradTens = np.zeros((2,3,3),dtype=self.dtype)
#            for inod in range(self.Nnod):
#                gradx = np.array([dN_[0,inod],\
#                                  0.0,dN_[1,inod]])
#                gradxx = np.einsum('ij,k',self.ident1,gradx)
#                gradTens += gradxx
#                if math.fabs(self.x_[0]) > 1.0e-14:
#                    gradx = np.array([0,self.Ns_[0][inod]/self.x_[0],0.0])
#                    gradxx = np.einsum('ij,k',self.ident2,gradx)
#                    gradTens += gradxx
#            for i in range(self.Nnod):
#                np.einsum('i,ijk',self.Nodes[i].getU().toNumpy()[0:2],\
#                          gradTens,out=self.temp_graduM_)
#                self.graduM_ += self.temp_graduM_
            
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
        
        kt33 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotN[:,jnod])
#        kt33 = np.einsum('i,i',self.rotN[:,inod],self.rotN[:,jnod])
        kt33 /= self.JF*self.material.mu0
#        kt33 /= self.material.mu0
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
        
    def calculateRe(self, R, inod, t):
        R.fill(0.0)
        re = self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R += re
        

        

class AxiSymMagneticBoundary(AE.AxisymmetricStaticBoundary,AxiMechElement,\
                             FB.StraightBoundary1D):
    def __init__(self,Nodes,pd,basisFunction,nodeOrder,intData,intSingData,\
                 normv,ide):
        AE.AxisymmetricStaticBoundary.__init__(self,Nodes,pd,basisFunction,\
                                               nodeOrder,intData,intSingData,\
                                               normv,ide)
        self.rotNs = []
        for i in range(self.intData.getNumberPoint()):
            self.rotNs.append(np.empty((self.Ndim,self.Nnod),self.dtype))
            
        self.rotNsx = []
        for i in range(self.intSingData.getNumberPoint()):
            self.rotNsx.append(np.empty((self.Ndim,self.Nnod),self.dtype))
            
        self.rotA = np.empty(self.Ndim,self.dtype)
        self.rotAx = np.empty(self.Ndim,self.dtype)
        self.RiLambd = []
        for i in range(self.Nnod):
            self.RiLambd.append(ia.zeros(self.Ndof,self.dtype))
        
        self.kt = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(self.Ndof,self.dtype)
        self.tangent = -np.cross(np.array(np.array([0.0,0.0,1.0])),\
                                         [self.normv[0],self.normv[1],0.0],\
                                         )[0:2]
        
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
        self.graduMx_ = np.zeros((3,3),dtype = self.dtype)
        self.temp_graduM_ = np.zeros((3,3),dtype = self.dtype)
        self.E_eng = np.zeros(4,dtype=self.dtype)
        self.E_eng_temp_ = np.zeros(4,dtype=self.dtype)
        
        self.sFg = np.empty((3,3),self.dtype)
        self.sFmg = np.empty((3,3),self.dtype)
        self.sEg = np.empty((3,3),self.dtype)
        self.sSg = np.empty((3,3),self.dtype)
        self.sFgx = np.empty((2,2),dtype=self.dtype)
        self.sFmgx = np.empty((2,2),dtype=self.dtype)
        self.sCgx = np.empty((2,2),dtype=self.dtype)
        
        self.KtT = []
        self.KtS = []
        for i in range(self.Nnod):
            self.KtT.append([])
            self.KtS.append([])
            for j in range(self.Nnod):
                self.KtT[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
                self.KtS[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
        
        self.tangent /= np.linalg.norm(self.tangent)
        self.deformed = True
#        self.grgrG = np.zeros((2,2),dtype = self.dtype)
        
    def prepareElement(self):
        AxiMechElement.prepareElement(self)
        self.gradXX = []
        x = np.empty(self.Ndim,self.dtype)
        for ig in range(self.intSingData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Nsx_[ig])
            self.gradXX.append(np.zeros((self.Nnod,2,3,3),dtype = self.dtype))
            for inod in range(self.Nnod):
                gradx = np.array([self.dNsx_[ig][0,inod],\
                                  0.0,self.dNsx_[ig][1,inod]])
                gradxx = np.einsum('ij,k',self.ident1,gradx)
                self.gradXX[ig][inod,:,:,:] += gradxx
                if math.fabs(x[0]) > 1.0e-14:
                    gradx = np.array([0,self.Nsx_[ig][inod]/x[0],0.0])
                    gradxx = np.einsum('ij,k',self.ident2,gradx)
                    self.gradXX[ig][inod,:,:,:] += gradxx
                    
        for ig in range(self.intSingData.getNumberPoint()):
            x.fill(0.0)
            self.getX(x,self.Nsx_[ig])
            for i in range(self.Nnod):
                self.rotNsx[ig][0,i]= -self.dNsx_[ig][1,i]
                if math.fabs(x[0]) < 1.0e-14:
                    self.rotNsx[ig][1,i] = 0.0
                    continue
                try:
                    self.rotNsx[ig][1,i]= self.Nsx_[ig][i]/x[0] +\
                    self.dNsx_[ig][0,i]
                except IndexError:
                    print('error here')
                    
                    
    def getFactor(self):
        """
        Return: factor for integration
        """
        return AE.AxisymmetricStaticBoundary.getFactor(self);
        
    def postCalculateF(self, N_, dN_, factor, res):
        idofA = 2
        idofJ = 3
        r = self.x_[0]
        k1 = self.u_[idofA]*\
        (self.normv[0]*self.gradG[0]+self.normv[1]*self.gradG[1])
        k1 *= factor*r
        k1 += self.u_[idofA]*self.G*self.normv[0]*factor
        k2 = self.u_[idofJ]*self.G
        k2 *= factor*r
        res += (k1 + k2)
                    
    def subCalculateFES(self):
        """
        calculate deformation gradient F
        """
        self.rotAx[0] = -self.gradux_[1,2]
        if math.fabs(self.xx_[0]) > 1.0e-14:
            self.rotAx[1] = self.ux_[2]/self.xx_[0] + self.gradux_[0,2]
        else:
            self.rotAx[1] = 0.0
        np.copyto(self.sFg,self.graduMx_)
        self.sFg += self.ident
        self.sFgx[0,0] = self.sFg[0,0]
        self.sFgx[0,1] = self.sFg[0,2]
        self.sFgx[1,0] = self.sFg[2,0]
        self.sFgx[1,1] = self.sFg[2,2]
        self.sJF = self.sFg[0,0]*self.sFg[2,2]-self.sFg[0,2]*self.sFg[2,0]
        self.sJF *= self.sFg[1,1]
        self.sFmg.fill(0.0)
        self.sFmg[0,0] = self.sFg[2,2]*self.sFg[1,1]/self.sJF
        self.sFmg[2,2] = self.sFg[0,0]*self.sFg[1,1]/self.sJF
        self.sFmg[0,2] = -self.sFg[0,2]*self.sFg[1,1]/self.sJF
        self.sFmg[2,0] = -self.sFg[2,0]*self.sFg[1,1]/self.sJF
        self.sFmg[1,1] = self.sFg[0,0]*self.sFg[2,2]-self.sFg[2,0]*self.sFg[0,2]
        self.sFmg[1,1] /= self.sJF
        self.sFmgx[0,0] = self.sFmg[0,0]
        self.sFmgx[1,1] = self.sFmg[2,2]
        self.sFmgx[0,1] = self.sFmg[0,1]
        self.sFmgx[1,0] = self.sFmg[1,0]
        self.sCg = np.dot(self.sFg.T,self.sFg)
        self.sCgx[0,0] = self.sCg[0,0]
        self.sCgx[0,1] = self.sCg[0,2]
        self.sCgx[1,0] = self.sCg[2,0]
        self.sCgx[1,1] = self.sCg[2,2]
        self.sEg = self.sCg - self.ident
        self.sEg *= 0.5
        np.einsum('ijkl,kl',self.material.Cmat,self.sEg,out=self.sSg)               
    
        
    def getBasisX(self):
        AE.AxisymmetricStaticBoundary.getBasisX(self)
        self.rotNX = self.rotNsx[self.ig]
        self.gradTensX = self.gradXX[self.ig]
#        self.Bmat = self.BmatA[self.ig]
    
    def getGradUX(self, gradu, data, dN_ = None):
        FE.Element.getGradU(self,gradu,data,dN_)
#        self.getGradUP(gradu,dN_)
#        self.graduMx_.fill(0.0)
#        for i in range(self.Nnod):
#            np.einsum('i,ijk',self.Nodes[i].getU().toNumpy()[0:2],\
#                      self.gradTensX[i,:,:,:],out=self.temp_graduM_)
#            self.graduMx_ += self.temp_graduM_
        
    
    def calculateGreen(self, x, xp):
#        if np.allclose(x,xp,rtol=1.0e-13) or math.fabs(xp[0])<1.0e-14 or\
        if math.fabs(x[0])<1.0e-14:
            self.G = 0.0
            self.gradG.fill(0.0)
            self.grgrG.fill(0.0)
            raise FB.SingularPoint
        r = x[0]
        rp = xp[0]
        z = x[1]
        zp = xp[1]
        self.G = self.Gfunc(rp,zp,r,z)
#        if np.isnan(self.G):
#            print('nan here')
        self.gradG[0] = self.Gdr(rp,zp,r,z)
        self.gradG[1] = self.Gdz(rp,zp,r,z)
#        self.gradG0[0] = self.Gdr0(rp,zp,r,z)
#        self.gradG0[1] = self.Gdz0(rp,zp,r,z)
#        self.grgrG[0,0] = self.Gdrr(rp,zp,r,z)
#        self.grgrG[0,1] = self.Gdrz(rp,zp,r,z)
#        self.grgrG[1,0] = self.Gdzr(rp,zp,r,z)
#        self.grgrG[1,1] = self.Gdzz(rp,zp,r,z)
#        self.gr0grG[0,0] = self.Gdr0r(rp,zp,r,z)
#        self.gr0grG[0,1] = self.Gdr0z(rp,zp,r,z)
#        self.gr0grG[1,0] = self.Gdz0r(rp,zp,r,z)
#        self.gr0grG[1,1] = self.Gdz0z(rp,zp,r,z)
        
    def derivativeJinvX(self, jnod):
        kt = np.einsum('mn,knm',self.sFmg,self.gradTensX[jnod,:,:,:])
        kt *= -1.0/self.sJF
        return kt
        
    def derivativeC1(self, jnod, dN_):
        if np.fabs(self.normv[0])<1.0e-13:
            return np.array([dN_[0,jnod],0.0])
        else:
            return np.array([0.0,dN_[1,jnod]])
        
    def derivativeC2(self, jnod, dN_):
        if np.fabs(self.normv[0])<1.0e-13:
            return np.array([self.u_[1]*dN_[0,jnod],\
                             dN_[0,jnod]*self.sFg[0,0]])
        else:
            return np.array([self.u_[0]*dN_[1,jnod],\
                             dN_[1,jnod]*self.sFg[1,1]])
    
    def derivativeFm11(self, jnod):
        k1 = self.derivativeJinv(jnod)
        k1 *= self.Fg[0,0]*self.Fg[2,2]-self.Fg[0,2]*self.Fg[2,0]
        k21 = self.dN_[0,jnod]*self.Fg[2,2]-self.dN_[1,jnod]*self.Fg[2,0]
        k22 = self.Fg[0,0]*self.dN_[1,jnod]-self.Fg[0,2]*self.dN_[0,jnod]
        k1[0] += k21/self.JF
        k1[1] += k22/self.JF
        return k1
    
    def multiplyFderiv(self,b,c,inod,jnod):
        kt1 = np.einsum('pkq,q',self.gradTens[jnod,:,:,:],c)
        kt2 = np.einsum('jik,i',self.gradTens[inod,:,:,:],b)
        kt = np.dot(kt2,kt1)
        
        return kt
    
    def calculateKLinear(self, K, i, j, t):
        K[2,3] = self.N_[i]*self.N_[j]
        K[2,3] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
        
    def subCalculateKLinear(self, K, element, i, j):
        pass
        
    def calculateK(self, K, inod, jnod, t):
        r = self.x_[0]
        
        normv = self.normv
        nB = np.dot(normv,self.rotA)
        nFm = np.dot(normv,self.Fmgx)
        Fn = np.dot(self.Fgx,normv)
#        Fmt = np.dot(self.Fmgx,self.tangent)
        Fmt = np.dot(self.tangent,self.Fmgx);
        Jcur = self.u_[3]
        Fg = self.Fg
        Fpp = Fg[1,1];
        
        d2Fpp_du = np.zeros(2,dtype= self.dtype);
        d2Fpp_du[0] = 2.0*self.N_[jnod]/r*Fpp;
        dJinv = np.zeros(2,dtype = self.dtype)
        dJinv[0] = (self.dN_[0,jnod]*Fg[2,2]-self.dN_[1,jnod]*Fg[2,0])*Fg[1,1]
        dJinv[0] += (Fg[0,0]*Fg[2,2]-Fg[2,0]*Fg[0,2])*self.N_[jnod]/r
        dJinv[1] = (Fg[0,0]*self.dN_[1,jnod]-Fg[0,2]*self.dN_[0,jnod])*Fg[1,1]
        dJinv /= -self.JF*self.JF
        dnB_dA = np.dot(normv,self.dN_[:,jnod])
        dnFm_du = np.zeros((2,2),dtype=self.dtype)
        dnFm_du[0,0] = (normv[0]*Fg[2,2]-normv[1]*Fg[2,0])*self.N_[jnod]/r
        dnFm_du[0,1] = (normv[0]*self.dN_[1,jnod]-normv[1]*self.dN_[0,jnod])
        dnFm_du[0,1] *= Fg[1,1]
        dnFm_du[1,0] = -(normv[0]*self.dN_[1,jnod]-normv[1]*self.dN_[0,jnod])
        dnFm_du[1,0] *= Fg[1,1]
        dnFm_du[1,0] -= (normv[0]*Fg[0,2]-normv[1]*Fg[0,0])*self.N_[jnod]/r
        dnFm_du /= self.JF
        dnFm_du += np.einsum('i,j',nFm,dJinv)
        
        dFn_du = np.array([[dnB_dA,0],[0,dnB_dA]])
        
        dFmt_du = np.zeros((2,2),dtype=self.dtype)
        dFmt_du[0,0] = -(normv[0]*Fg[2,0]+normv[1]*Fg[2,2])*self.N_[jnod]/r
        dFmt_du[0,1] = -(dnB_dA)*Fg[1,1]
        dFmt_du[1,0] = (normv[0]*Fg[0,0]+normv[1]*Fg[0,2])*self.N_[jnod]/r
        dFmt_du[1,0] += dnB_dA*Fg[1,1]
        dFmt_du /= self.JF
        dFmt_du += np.einsum('i,j',Fmt,dJinv)
        
        t1 = 0.5*(nB*nB+Jcur*Jcur*Fpp*Fpp)*nFm
        t2 = nB*nB*Fn
        ktxy = np.einsum('i,j',-t1+t2,dJinv)
        ktxy -= 0.5*(nB*nB+Jcur*Jcur*Fpp*Fpp)/self.JF*dnFm_du
        ktxy -= 0.5*Jcur*Jcur/self.JF*np.einsum('i,j',nFm,d2Fpp_du);
        ktxy += nB*nB/self.JF*dFn_du
        ktxy += nB*Jcur*dFmt_du
        
        ktxy *= 1.0/self.material.mu00
        ktxy *= self.N_[inod]*self.getFactor()*2.0*np.pi*r
        
        ktxy *= -1.0
        
        K[0,0] += ktxy[0,0]
        K[0,1] += ktxy[0,1]
        K[1,0] += ktxy[1,0]
        K[1,1] += ktxy[1,1]
        
        kxy2 = -nB*dnB_dA*nFm/self.JF
        kxy2 += nB*dnB_dA*Fn/self.JF
        kxy2 += dnB_dA*Jcur*Fmt
        
        kxy2 *= 1.0/self.material.mu00
        kxy2 *= self.N_[inod]*self.getFactor()*2.0*np.pi*r
        
        kxy2 *= -1.0
        
        K[0,2] += kxy2[0]
        K[1,2] += kxy2[1]
        
        kxy3 = -Jcur*self.N_[jnod]*Fpp*Fpp*nFm/self.JF
        kxy3 += nB*self.N_[jnod]*Fmt
        
        kxy3 *= 1.0/self.material.mu00
        kxy3 *= self.N_[inod]*self.getFactor()*2.0*np.pi*r
        
        kxy3 *= -1.0
        
        K[0,3] += kxy3[0]
        K[1,3] += kxy3[1]
        
        k30 = 0.5*self.N_[inod]*self.derivativeFm11(jnod)*self.u_[2]
        k30 *= self.getFactor()*r*2.0*np.pi/self.material.mu00
        K[3,0] -= k30[0]
        K[3,1] -= k30[1]
        k32 = 0.5*self.N_[inod]*self.N_[jnod]*self.Fmg[1,1]*self.getFactor()
        k32 *= r*2.0*np.pi/self.material.mu00
        K[3,2] -= k32
        
        
    def calculateR(self, R, inod, t):
        nB = np.dot(self.normv,self.rotA)
        nFm = np.dot(self.normv,self.Fmgx)
        Fn = np.dot(self.Fgx,self.normv)
#        Fmt = np.dot(self.Fmgx,self.tangent)
        Fmt = np.dot(self.tangent,self.Fmgx);
        Fpp = self.Fg[1,1];
        Jcur = self.u_[3]
        t1 = 0.5/self.JF*(nB*nB+Jcur*Jcur*Fpp*Fpp)*nFm
        t2 = nB*nB/self.JF*Fn
        t2 += nB*Jcur*Fmt
        
        ri = -t1 + t2        
        ri *= 1.0/self.material.mu00        
        ri *= self.N_[inod]*self.getFactor()*2.0*np.pi*self.x_[0]
        
        ri *= -1.0
        
        R[0] += ri[0]
        R[1] += ri[1]
        
#        if np.fabs(self.Fmg[1,1]-1.0)>1.0e-13:
#            print('raise exception')
#            raise Exception;
        
        factor = self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
        R[3] -= 0.5*self.N_[inod]*self.u_[2]*self.Fmg[1,1]*factor
        
#    def subCalculateKLinear(self, K, element, i, j):
#        #if np.allclose(element.xx_[0],0.0,rtol = 1.0e-14):
#        #    K.fill(0.0)
#        #    return
#        wfac = self.getFactor()
#        wfact = element.getFactorX(element.detJ)
#        wfacx = element.getFactorXR(element.detJ)
#        K[1,0] = self.N_[i]*element.Nx_[j]*\
#        (element.normv[0]*self.gradG[0]+element.normv[1]*self.gradG[1])
#        K[1,0] *= wfac*wfacx
#        K[1,0] += self.N_[i]*element.Nx_[j]*self.G*\
#        element.normv[0]*wfac*wfact/element.xx_[0]
#        K[1,1] = self.N_[i]*element.Nx_[j]*self.G
#        K[1,1] *= wfac*wfact    
        
    def subCalculateK(self, K, element, inod, jnod):
        K.fill(0.0)
        r0 = self.x_[0]
        r = element.xx_[0]
        wfac = self.getFactor()*r0*2.0*np.pi
        wfact = element.getFactorX(element.detJ)*r
        wfacx = element.getFactorXR(element.detJ)*r
        
        Frr = element.sFg[0,0]
        Frz = element.sFg[0,2]
        Fzr = element.sFg[2,0]
        Fzz = element.sFg[2,2]
        Fpp = element.sFg[1,1]
        
#        if np.fabs(Frr-1.0)>1.0e-13 or np.fabs(Fzz-1.0)>1.0e-13 or\
#        np.fabs(Fpp-1.0)>1.0e-13 or np.fabs(Fzr)>1.0e-13 or np.fabs(Frz)>1.0e-13:
#            print('raise exception')
#            raise Exception
        
        dFrr_ur = element.dNx_[0,jnod]
        dFrz_ur = element.dNx_[1,jnod]
        dFzr_uz = element.dNx_[0,jnod]
        dFzz_uz = element.dNx_[1,jnod]
        dFpp_ur = element.Nx_[jnod]/r
        
        k3x0 = -self.gradG[1]*element.normv[0]*dFrz_ur
        k3x0 += self.gradG[1]*element.normv[1]*dFrr_ur
        k3x1 = -self.gradG[0]*element.normv[1]*dFzr_uz
        k3x1 += self.gradG[0]*element.normv[0]*dFzz_uz
        
        k3x1 -= (element.normv[1]*dFzr_uz-element.normv[0]*dFzz_uz)/r*self.G
        
        k3x0 *= wfacx*wfac*self.N_[inod]*element.ux_[2]
        k3x1 *= wfacx*wfac*self.N_[inod]*element.ux_[2]
        
        K[3,0] = k3x0
        K[3,1] = k3x1
        
#        k3xy0 = -self.grgrG[1,:]*(element.normv[0]*Frz-element.normv[1]*Frr)
#        k3xy0 *= wfacx*wfac*self.N_[inod]*element.Nx_[jnod]*element.ux_[2]
#        
#        K[3,0] += k3xy0[0]
#        K[3,1] += k3xy0[1]
#        
#        k3xy1 = -self.grgrG[0,:]*(element.normv[1]*Fzr-element.normv[0]*Fzz)
#        k3xy1 *= wfacx*wfac*self.N_[inod]*element.Nx_[jnod]*element.ux_[2]
#        
#        K[3,0] += k3xy1[0]
#        K[3,1] += k3xy1[1]
        
        k32 = -self.gradG[0]*element.normv[1]*Fzr
        k32 -= self.gradG[1]*element.normv[0]*Frz
        k32 += self.gradG[1]*element.normv[1]*Frr
        k32 += self.gradG[0]*element.normv[0]*Fzz
        
        k32 *= wfacx*wfac*self.N_[inod]*element.Nx_[jnod]
        
        k32 -= (element.normv[1]*Fzr-element.normv[0]*Fzz)/(r)*self.G*\
        self.N_[inod]*element.Nx_[jnod]*wfacx*wfac
        
        K[3,2] = k32
        
        k3y = element.ux_[3]*dFpp_ur*self.G
        k3y *= self.N_[inod]*wfact*wfac
        K[3,0] += k3y
        
#        k3Gxy = -element.ux_[2]*self.gradG/(r)
#        k3Gxy *= (element.normv[1]*Fzr-element.normv[0]*Fzz)*wfacx
#        k3Gxy += element.ux_[3]*self.gradG*wfact
#        k3Gxy *= self.N[inod]*element.Nx_[jnod]*wfac
#        
#        K[3,0] += k3Gxy[0]
#        K[3,1] += k3Gxy[1]
        
        k33 = element.Nx_[jnod]*Fpp*self.G
        k33 *= self.N_[inod]*wfact*wfac
        K[3,3] = k33
        
        K /= self.material.mu00
        
    def subCalculateR(self, R, element, inod):
        r0 = self.x_[0]
        r = element.xx_[0]
        wfac = self.getFactor()*r0*2.0*np.pi
        wfact = element.getFactorX(element.detJ)*r
        wfacx = element.getFactorXR(element.detJ)*r
        Frr = element.sFg[0,0]
        Frz = element.sFg[0,2]
        Fzr = element.sFg[2,0]
        Fzz = element.sFg[2,2]
        Fpp = element.sFg[1,1]
        
        ri = self.gradG[0]*element.normv[1]*Fzr
        ri += self.gradG[1]*element.normv[0]*Frz
        ri *= -1.0
        ri += element.normv[1]*self.gradG[1]*Frr
        ri += element.normv[0]*self.gradG[0]*Fzz
        ri -= self.G/(r)*(element.normv[1]*Fzr-element.normv[0]*Fzz)
        ri *= element.ux_[2]
        ri *= wfacx
        
        ri += element.ux_[3]*Fpp*self.G*wfact
        
        ri *= self.N_[inod]*wfac/self.material.mu00
        
        R[3] += ri
        
#    def subCalculateKLinear(self, K, element, i, j):
#        #if np.allclose(element.xx_[0],0.0,rtol = 1.0e-14):
#        #    K.fill(0.0)
#        #    return
#        idofA = 2
#        idofJ = 3
#        r0 = self.x_[0]
#        r = element.xx_[0]
#        wfac = self.getFactor()*r0
#        wfact = element.getFactorX(element.detJ)
#        wfacx = element.getFactorXR(element.detJ) 
#        
#        K[idofJ,idofA]=self.N_[i]*element.Nx_[j]
#        K[idofJ,idofA]*=np.dot(element.normv,self.gradG)
#        
#        K[idofJ,idofA] += self.N_[i]*element.Nx_[j]*self.G*element.normv[0]/r
#        K[idofJ,idofA]*= wfac*wfacx*r
#        
#        K[idofJ,idofJ]=self.N_[i]*element.Nx_[j]*self.G
#        K[idofJ,idofJ]*= wfac*wfact*r
#        
#        K /= self.material.mu00
#        K *= 2.0*np.pi
#    
#    def calculateKLinear(self, K, i, j, t):
#        K[2,3] = self.N_[i]*self.N_[j]
#        K[2,3] *= self.getFactor()*self.x_[0]*2.0*np.pi/self.material.mu00
#        K[3,2] = -self.N_[i]*self.N_[j]*0.5
#        K[3,2] *= self.getFactor()*self.x_[0]*2.0*np.pi
#        K[3,2] /= self.material.mu00
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None,\
             deformed = False, deformed_factor=1.0 ):
        return FE.StandardElement.plot(self,fig,col,fill_mat,number,\
                                deformed,deformed_factor)

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
        idofA = 2
        idofJ = 3
        r = self.x_[0]
        k1 = self.u_[idofA]*\
        (self.normv[0]*self.gradG[0]+self.normv[1]*self.gradG[1])
#        k1 *= factor*r
        k1 *= factor
        k1 += self.u_[idofA]*self.G*self.normv[0]*factor/r
        k2 = self.u_[idofJ]*self.G
#        k2 *= factor*r
        k2 *= factor
        res += (k1 + k2)
        
    def subCalculateKLinear(self, K, element, i, j):
        #if np.allclose(element.xx_[0],0.0,rtol = 1.0e-14):
        #    K.fill(0.0)
        #    return
        idofA = 2
        idofJ = 3
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
        K[2,3] = self.N_[i]*self.N_[j]
        K[2,3] *= self.getFactor()*self.x_[0]*2.0*np.pi/self.material.mu00
        K[3,2] = self.N_[i]*self.N_[j]*0.5
        K[3,2] *= self.getFactor()*self.x_[0]*2.0*np.pi
        K[3,2] /= self.material.mu00

class MeshPostProcessingAxisymmetric(FM.MeshWithBoundaryElement):
    def findBoundaryNodes(self):        
        self.BoundaryNodesUp = []
        self.BoundaryNodesDw = []
        tempNds = self.BoundaryElements[40:43]
        BoundaryDw =self.BoundaryElements[0:20]+tempNds+\
        self.BoundaryElements[20:40] + self.BoundaryElements[43:-1]
        for be in BoundaryDw:
            for n in be.Nodes:
                if n.getX()[1] > -0.005:
                    if n not in self.BoundaryNodesUp:
                        self.BoundaryNodesUp.append(n)
                else:
                    if n not in self.BoundaryNodesDw:
                        self.BoundaryNodesDw.append(n)
                        
    def isInsdeBodies(self, x):
        if x[1] > -0.005:
            return pnpoly(len(self.BoundaryNodesUp),self.BoundaryNodesUp,x[0],x[1])
        return pnpoly(len(self.BoundaryNodesDw),self.BoundaryNodesDw,x[0],x[1])
    
    def getValue(self, x, val='u',intDat = None):
#        if self.isInsdeBodies(x):
        for i,e in enumerate(self.Elements):
            if i >= (self.Ne-self.NBe):
                continue
            try:
                xi = e.getXi(x)
                return e.postCalculate(xi, val)
            except FM.OutsideElement:
                continue
        if val != 'u':
            raise FM.OutsideMesh
            
        res = np.zeros(self.Nodes[0].Ndof,self.Nodes[0].dtype)
        for e in self.BoundaryElements:
            res += e.postCalculateX(x,intDat)
            
        return res

def pnpoly(nvert, vert, testx,testy):
    i= 0
    j = nvert - 1
    c = False
    while i < nvert:
        if ((vert[i].getX()[1]>testy) != (vert[j].getX()[1]>testy)) and\
        (testx < (vert[j].getX()[0]-vert[i].getX()[0]) * (testy-vert[i].getX()[1]) / (vert[j].getX()[1]-vert[i].getX()[1]) + vert[i].getX()[0]):
            c = not c
        j = i
        i += 1
    return c                    
    


Ndof = 4             
tOrder = 0
Ng = [3,3]
numberSteps = 2
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(6, 1, idat.Gaussian1D)
intDatC = idat.GaussianQuadratureFile('../Gauss_Data_2.dat')
intSingDat = SI.SingularGaussian1D(24, intDatB.xg,\
SI.Gaussian_1D_Pn_Log, SI.Gaussian_1D_Pn_Log_Rat)
#intSingDat = idat.GaussianQuadrature(12, 1, idat.Gaussian1D)

condt = np.array([np.cos(np.pi/2.0+1.0e-2*np.pi),\
                  np.sin(np.pi/2.0+1.0e-2*np.pi)])
#condt = np.array([1.0e5,0.0])

def loadfunc(x, t):
    return np.array([0.0,0.0,1250.0/0.01/0.01,0.0])

def loadfuncG(x, t):
    return np.array([0.0,-9.8*7.8e3,0.0,0.0])

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh_BI():
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
    polys[2].setDivisionEdge13(30)
    polys[2].setDivisionEdge24(1)
    
#    polys[4].setDivisionEdge13(8)
#    polys[4].setDivisionEdge24(1)
    
    mat1 = LinearMechanicMaterial(2.1e9, 0.3, 7.8e3,1000.0,0)
    mat2 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,1)
    mat3 = LinearMechanicMaterial(0.0, 0.0, 1.0,100.0,2)
    mat4 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,3)
    
    polys[0].setMaterial(mat3)
    polys[1].setMaterial(mat2)
    polys[2].setMaterial(mat1)
#    polys[3].setMaterial(mat2)
#    polys[4].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:
#        n.setConstraint(False, 0.0, 0)
#        n.setConstraint(False, 0.0, 1)
#        if math.fabs(n.getX()[0]-R)<1.0e-14 and math.fabs(n.getX()[1]-H*0.5)<1.0e-14:
#        if math.fabs(n.getX()[0]-R)<1.0e-14 :
#            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False, 0.0, 1)
#            n.setLoad(-condt[0],0)
#            ncenter = n
#            n.setLoad(-1.0e6,0)
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
#        n.setConstraint(False,0.0,0)
#        n.setConstraint(False,0.0,1)
        if math.fabs(n.getX()[0]-R)<1.0e-14 :
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            
#        if n.getX()[1]<-H*10-1.0e-14 and n.getX()[1]>-H*20+1.0e-14 and\
#        n.getX()[0]<2.1*R-1.0e-14:
#            n.setConstraint(False, 0.0,3)
#            
#        if n.getX()[1]<H*21-1.0e-14 and n.getX()[1]>H*11+1.0e-14 and\
#        n.getX()[0]<2.1*R-1.0e-14:
#            n.setConstraint(False, 0.0,3)
#            
#        if n.getX()[1]<H-1.0e-14 and n.getX()[1]>1.0e-14 and\
#        n.getX()[0]<R-1.0e-14:
#            n.setConstraint(False, 0.0,3)
            
        if n.getX()[1] > H*1.5 or n.getX()[1] < -0.5*H:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
            
        if math.fabs(n.getX()[0])<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 2)
            n.setConstraint(False, 0.0, 3)
#            n.setConstraint(False,0.0,2)
#        if math.fabs(n.getX()[1])<1.0e-14 or\
#        math.fabs(n.getX()[1]-H)<1.0e-14:
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            
#    for n in nodesx:
#        if math.fabs(n.getX()[0]-R)<1.0e-14:
#            n.friendOF(ncenter,0)
            
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiMechElement(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat,condt))
#        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
#        nodeOrder,m,intDat))
        if m.idx == 1:
            elements[-1].setBodyLoad(loadfunc)
#        if m.idx == 0:
#            elements[-1].setBodyLoad(loadfuncG)
#        elements[-1].setBodyLoad(loadfunc)
    
#    mesh =  FM.MeshWithBoundaryElement()
    mesh = MeshPostProcessingAxisymmetric()
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
        elementBs.append(AxiSymMagneticBoundary(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(mat2)
        
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
        elementBs[-1].linear=False
    

    mesh.addBoundaryElements(elementBs)   
    
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 3)
        
    return mesh

def create_mesh_FE():
    H = 0.001
    R = 0.1
    nodes = []
    nodes.append(FN.Node([0,-3*H*10],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R,-3*H*10],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R*2,-3*H*10],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([R*2+0.1*R,-3*H*10],Ndof,timeOrder = tOrder))    
    nodes.append(FN.Node([R*3,-3*H*10],Ndof,timeOrder = tOrder)) 

    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [10*H,10*H,10*H,H,10*H]
    
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
        
    polys = geo.getPolygons()
    
    for i in range(5):  
        polys[i].setDivisionEdge13(10)
        polys[i].setDivisionEdge24(1)
    for i in range(5,10):
        polys[i].setDivisionEdge13(10)
        polys[i].setDivisionEdge24(1)
    for i in range(15,20):
        polys[i].setDivisionEdge13(8)
        polys[i].setDivisionEdge24(1)
        
    polys[2].setDivisionEdge24(2)
    polys[7].setDivisionEdge24(2)
    polys[12].setDivisionEdge24(2)
    polys[17].setDivisionEdge24(2)
    
    polys[4].setDivisionEdge24(2)
    polys[9].setDivisionEdge24(2)
    polys[14].setDivisionEdge24(2)
    polys[19].setDivisionEdge24(2)
    
#    polys[4].setDivisionEdge13(8)
#    polys[4].setDivisionEdge24(1)
    
    mat1 = LinearMechanicMaterial(2.1e9, 0.3, 7.8e3,1000.0,0)
    mat2 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,2)
    mat3 = LinearMechanicMaterial(0.0, 0.0, 1.0,100.0,1)
    mat4 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,3)
    
    for po in polys:
        po.setMaterial(mat2)
    polys[1].setMaterial(mat3)
    polys[6].setMaterial(mat3)
    polys[3].setMaterial(mat1)
    polys[11].setMaterial(mat4)
#    polys[3].setMaterial(mat2)
#    polys[4].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:
#        n.setConstraint(False, 0.0, 0)
#        n.setConstraint(False, 0.0, 1)
        if n.getX()[0]>R - 1.0e-10*R :
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)

            
        if n.getX()[1] > H+1.0e-13 or n.getX()[1] < -1.0e-13:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
            
        if math.fabs(n.getX()[0])<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 2)
            n.setConstraint(False, 0.0, 3)
            
#    for n in nodesx:
#        if math.fabs(n.getX()[0]-R)<1.0e-14:
#            n.friendOF(ncenter,0)
            
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiMechElement(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat,condt))
        if m.idx == 3:
            elements[-1].setBodyLoad(loadfunc)
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
        elementBs[-1].setMaterial(mat2)
        
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
            n.setConstraint(False, 0.0, 3)
        
    return mesh

mesh_BI = create_mesh_BI()
mesh_BI.generateID()

center_nodes_num_BI = [132,133,134,140,141,146,147,152,153,158,159,164,165,170,\
                    171,176,177,182,183,188,189,194,195,200,201,206,207,212,\
                    213,218,219,224,225,230,236,237,242,243,248,249,254,255,\
                    260,261,266,267,272,273,278,279,284,285,290,291,296,297,\
                    302,303,308,309]
center_nodes_BI = [mesh_BI.Nodes[i] for i in center_nodes_num_BI]
X_BI = [n.getX().tolist()[0] for n in center_nodes_BI]

mesh_FE = create_mesh_FE()
mesh_FE.generateID()

center_nodes_num_FE = [230,229,226,225,222,221,216,217,214,213,210,209,206,\
                       205,202,201,198,197,194,193,192]
center_nodes_FE = [mesh_FE.Nodes[i] for i in center_nodes_num_FE]
X_FE = [n.getX().tolist()[0] for n in center_nodes_FE]

#op_BI_500A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_500A.dat',"r")
#op_BI_500A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_500A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_500A.finishOutput();
#
#op_BI_750A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_750A.dat',"r")
#op_BI_750A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_750A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_750A.finishOutput();
#
#op_BI_1000A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_1000A.dat',"r")
#op_BI_1000A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_1000A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_1000A.finishOutput();
#
#op_BI_1250A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_1250A.dat',"r")
#op_BI_1250A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_1250A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_1250A.finishOutput();
#
op_BI_1500A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_1500A.dat',"r")
op_BI_1500A.updateToMesh(mesh_BI,istep=1)
Uz_BI_1500A = [n.getU().tolist()[1] for n in center_nodes_BI]
op_BI_1500A.finishOutput();


#op_BI_2000A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_2000A.dat',"r") 
#op_BI_2000A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_2000A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_2000A.finishOutput(); 

#op_BI_5000A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_BI_5000A.dat',"r") 
#op_BI_5000A.updateToMesh(mesh_BI,istep=1)
#Uz_BI_5000A = [n.getU().tolist()[1] for n in center_nodes_BI]
#op_BI_5000A.finishOutput(); 

#for n in mesh_BI.Nodes:
#    if (n.getU()[0] != 0):
#        n.getU()[2] *= n.getX()[0]/n.getU()[0]
#    if (n.getX()[0]) > 1.0e-14:
#        n.getU()[3] *= n.getU()[0]/n.getX()[0]
#    n.getX()[0] += n.getU()[0]
#    n.getX()[1] += n.getU()[1]
#
#for e in mesh_BI.Elements:
#    e.calculateBoundingBox()
    
mesh_BI.findBoundaryNodes()

#op_FE_500A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_FE_500A.dat',"r")
#op_FE_500A.updateToMesh(mesh_FE,istep=1)
#Uz_FE_500A = [n.getU().tolist()[1] for n in center_nodes_FE]
#op_FE_500A.finishOutput();
#
#op_FE_750A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_FE_750A.dat',"r")
#op_FE_750A.updateToMesh(mesh_FE,istep=1)
#Uz_FE_750A = [n.getU().tolist()[1] for n in center_nodes_FE]
#op_FE_750A.finishOutput();
#
#op_FE_1000A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_FE_1000A.dat',"r")
#op_FE_1000A.updateToMesh(mesh_FE,istep=1)
#Uz_FE_1000A = [n.getU().tolist()[1] for n in center_nodes_FE]
#op_FE_1000A.finishOutput();
#
#op_FE_1250A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_FE_1250A.dat',"r")
#op_FE_1250A.updateToMesh(mesh_FE,istep=1)
#Uz_FE_1250A = [n.getU().tolist()[1] for n in center_nodes_FE]
#op_FE_1250A.finishOutput();
#
#op_FE_1500A = FO.StandardFileOutput('/media/haiau/Work/new_results/result_FE_1500A.dat',"r")
#op_FE_1500A.updateToMesh(mesh_FE,istep=1)
#Uz_FE_1500A = [n.getU().tolist()[1] for n in center_nodes_FE]
#op_FE_1500A.finishOutput();    

for i,n in enumerate(mesh_FE.Nodes):
    res = mesh_BI.getValue(n.getX(),val='u',intDat=intDatC)
    if (n.getID()[0] >= 0):
        n.getU()[0] = res[0]
    if (n.getID()[1] >= 0):
        n.getU()[1] = res[1]
    if (n.getID()[2] >= 0):
        n.getU()[2] = res[2]
#    if n.getX()[0] > 1.0e-14:
#        n.getU()[2] /= (res[2]/n.getX()[0]+1.0)
#    if (n.getID()[0] >= 0):
#        n.getX()[0] += res[0]
#    if (n.getID()[1] >= 0):
#        n.getX()[1] += res[1]
    
X,Y,Za = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'u',idof=2)
X,Y,grA0 = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=2,idir=0)
X,Y,grA1 = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=2,idir=1)
X,Y,Zur = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'u',idof=0)
X,Y,durdr = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=0,idir=0)
X,Y,durdz = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=0,idir=1)
X,Y,duzdr = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=1,idir=0)
X,Y,duzdz = mesh_FE.meshgridValue([0.0,0.3,-0.03,0.011],(0.01,0.002),1.0e-7,\
                               'gradu',idof=1,idir=1)

Frr = durdr+1
Frz = durdz
Fzr = duzdr
Fzz = duzdz+1
Fpp = Zur
for i in range(Fpp.shape[0]):
    for j in range(Fpp.shape[1]):
        if X[i,j] > 0.0:
            Fpp[i,j] /= X[i,j]
Fpp += 1
Jdet = (Frr*Fzz-Frz*Fzr)*Fpp

B0 = -grA1
B1 = grA0 + Za/X
b0 = (Frr*B0 + Frz*B1)/Jdet
b1 = (Fzr*B0 + Fzz*B1)/Jdet
Bnorm = np.sqrt(B0*B0 + B1*B1)
bnorm = np.sqrt(b0*b0 + b1*b1)

mu0 = np.pi*4.0e-7

sigma_rr = 1.0/mu0*(b0*b0 - 0.5*bnorm)
sigma_zz = 1.0/mu0*(b1*b1 - 0.5*bnorm)
sigma_zr = 1.0/mu0*(b0*b1 - 0.5*bnorm)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if X[i,j]<0.2 and Y[i,j]<=-0.01 and Y[i,j]>=-0.02:
            sigma_rr[i,j] *= (2.0-100.0)/100.0
            sigma_zz[i,j] *= (2.0-100.0)/100.0
            sigma_zr[i,j] *= (2.0-100.0)/100.0
        if X[i,j]<=0.1 and Y[i,j]<=0.001 and Y[i,j]>=0.0:
            sigma_rr[i,j] *= (2.0-1000.0)/1000.0
            sigma_zz[i,j] *= (2.0-1000.0)/1000.0
            sigma_zr[i,j] *= (2.0-1000.0)/1000.0
        if X[i,j]<=0.21 and X[i,j]>=0.2 and Y[i,j]<=-0.01 and Y[i,j]>=-0.02:
            sigma_rr[i,j] *= 0
            sigma_zz[i,j] *= 0
            sigma_zr[i,j] *= 0

jet = pl.get_cmap('jet')

edges1x = np.array([0.0,0.21,0.21,0.0])
edges1y = np.array([-0.02,-0.02,-0.01,-0.01])

edges2x = np.array([0.0,0.1,0.1,0.0])
edges2y = np.array([0.0,0.0,0.001,0.001])

#pl.contour(X,Y,Za/(Zur/X+1),100)

cnt = pl.contourf(X,Y,sigma_rr,100,cmap=jet)
pl.plot(edges1x,edges1y,'-k',linewidth=0.5)
pl.plot(edges2x,edges2y,'-k',linewidth=0.5)

#pl.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
#pl.tick_params(axis='y',which='both',right=False,left=False,labelleft=False)

pl.gca().set_aspect('equal', adjustable='box')

for c in cnt.collections:
   c.set_edgecolor("face")
   
midlineNodes = []
#midlineGrad = []
for n in mesh_BI.Nodes:
    if (np.fabs(n.getX()[1]+0.01)<1.0e-15):
#        midlineGrad.append(mesh_FE.getValue(n.getX(),'gradu'))
        midlineNodes.append(n)
Amid1 = [n.getU()[2] for n in midlineNodes]
#Jmid1 = [-n[1][2] for n in midlineGrad]
Jmid1 = [n.getU()[3] for n in midlineNodes]
Xmid1 = [n.getX()[0] for n in midlineNodes]
   
#pl.savefig("/home/haiau/Pictures/sigma_rr_FE.pdf",bbox_inches='tight')    
#    
##import matplotlib.ticker as ticker
#fig,ax = pl.subplots()
#cb = pl.colorbar(cnt,ax=ax,orientation="horizontal")
#tick_locator = ticker.MaxNLocator(nbins=5)
#cb.locator = tick_locator
#cb.update_ticks()
#ax.remove()
#
#pl.savefig("/home/haiau/Pictures/sigma_rr_FE_colorbar.pdf",bbox_inches='tight')