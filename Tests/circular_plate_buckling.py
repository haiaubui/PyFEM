#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 15:22:23 2018

@author: haiau
"""

import math
import numpy as np
import pylab as pl
import Element.AxisymmetricElement as AE
import Element.QuadElement as QE
import Element.FEMElement as FE
import Mesh.FEMNode as FN
import Mesh.FEMMesh as FM
import InOut.FEMOutput as FO
import Material.Material as mat
import Algorithm.NewtonRaphson as NR
import Math.Solver as sv
import Math.IntegrationData as idat
import Mesh.MeshGenerator as mg
import Math.injectionArray as ia


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
        
        self.RiLambd = []
        for i in range(self.Nnod):
            self.RiLambd.append(ia.zeros(self.Ndof,self.dtype))
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
        
        self.KtT = []
        self.KtS = []
        for i in range(self.Nnod):
            self.KtT.append([])
            self.KtS.append([])
            for j in range(self.Nnod):
                self.KtT[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
                self.KtS[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
        
        
        self.kt = np.empty((2,2),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(2,self.dtype)
        self.condt = condt
#        self.linear = True
    
    def connect(self, *arg):
        AE.AxisymmetricQuadElement.connect(self, *arg)
#        KtT = arg[8]
#        KtS = arg[9]
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
        RiLamb = arg[8]
        for inod in range(self.Nnod):
            ids1 = self.Nodes[inod].getID()
            for idof in range(self.Ndof):
                id1 = ids1[idof]
                if id1 >= 0:
                    self.RiLambd[inod].connect(idof,RiLamb,id1)
    
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
#        self.rotA[0] = -self.gradu_[1,2]
#        if math.fabs(self.x_[0]) > 1.0e-14:
#            self.rotA[1] = self.u_[2]/self.x_[0] + self.gradu_[0,2]
#        else:
#            self.rotA[1] = 0.0
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
#        en = np.dot(a,b)
#        ktz1 = self.dN_[1,jnod]*self.Fg[0,2]*2.0*en/self.JF
#        ktz2 = self.dN_[1,jnod]*self.Fg[2,2]*2.0*en/self.JF
#        kt1 = np.array([ktz1,ktz2])
#        if not np.allclose(kt,kt1):
#            print('not close')
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
#        ktx = -np.einsum('nm,kmn',self.Fmg,self.gradTens[jnod,:,:,:])
#        ktu = np.dot(self.Fg,c)
#        ktp = np.einsum('i,j',b,ktu)
#        ktq = np.einsum('kij,ij',self.gradTens[inod,:,:,:],ktp)
#        kt = np.einsum('i,j',ktq,ktx)
#        kt = -np.einsum('lip,i,nm,kmn,pq,q',self.gradTens[inod,:,:,:],b,\
#                        self.Fmg,self.gradTens[jnod,:,:,:],self.Fg,c)
        kt1 = np.einsum('pkq,q',self.gradTens[jnod,:,:,:],c)
        kt2 = np.einsum('jik,i',self.gradTens[inod,:,:,:],b)
        kt = np.dot(kt2,kt1)
#        en = np.dot(b,c)
#        ktz1 = self.dN_[0,inod]*self.dN_[1,jnod]
#        ktz2 = self.dN_[1,inod]*self.dN_[1,jnod]
#        ktz1 *= en
#        ktz2 *= en
#        ktz1 *= np.dot(b,c)/self.JF
#        ktz2 *= np.dot(b,c)/self.JF
#        ktz *= np.dot(b,c)
        kt /= self.JF
        return kt
#        kt1 = np.array([[0.0,0.0],[ktz1,ktz2]])
#        if not np.allclose(kt.T,kt1):
#            print('not close')
#        return np.array([[0.0,0.0],[ktz1,ktz2]])
        
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
        kg = np.einsum('jim,kin,mn',self.gradTens[inod,:,:,:],\
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
        
        self.KtT[inod][jnod][0,0] += self.kt[0,0]
        self.KtT[inod][jnod][0,1] += self.kt[0,1]
        self.KtT[inod][jnod][1,0] += self.kt[1,0]
        self.KtT[inod][jnod][1,1] += self.kt[1,1]
        
        # Magnetics part-------------------------------------------------------
        kt33 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotN[:,jnod])
        kt33 /= self.JF*self.material.mu0
#        kt33 = np.einsum('i,i',self.rotN[:,inod],self.rotN[:,jnod])
#        kt33 /= self.material.mu0
        kt33 *= self.getFactor()
        
#        K[2,2] += kt33
        
        # Magnetic - Mechanics coupling
#        condt = np.array([self.rotA[0],0.0,self.rotA[1]])
#        rotN = np.array([self.rotN[0,inod],0.0,self.rotN[1,inod]])        
#        kt321 = self.multiplyCderiv(rotN,condt,jnod)
#        kt321 /= self.material.mu0*self.JF
#        en = np.einsum('i,ij,j',rotN,self.Cg,condt)/self.material.mu0
#        kt321 += en*self.derivativeJinv(jnod)
#        kt321 *= self.getFactor()
        
#        K[2,0] += kt321[0]
#        K[2,1] += kt321[1]
        
        #Magneto-mechanics part------------------------------------------------
#        condt = t*np.array([0.0,0.0,self.condt[1]])
        condt = np.array([0.0,0.0,self.condt[1]])
        
        # Derivative F^-1
#        en = np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        en = np.einsum('i,ij,j',condt,self.Cg,condt)
        en *= 0.5/(self.JF*self.material.mu0)
        kt1212 = self.multiplyFmderiv(inod,jnod)
        kt1212 *= en
        # Derivative C
        ktx = 0.5*self.multiplyCderiv(condt,condt,jnod)
#        ktx *= (2.0-self.material.mur)/(self.material.mu0)
        ktx *= 1.0/(self.material.mu0)
        kt1212 += np.einsum('kij,ij,l',self.gradTens[inod,:,:,:],self.Fmg,ktx)

        # Derivative 1/J
#        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)
        te = en * self.Fmg
        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
        ktx = np.einsum('kij,ij',self.gradTens[inod,:,:,:],te)
        ktx /= self.material.mu0
        kty = self.derivativeJinv(jnod)
        kt1212 += np.einsum('i,j',ktx,kty)
        
        # Derivative F
        ktxx = self.multiplyFderiv(condt,condt,inod,jnod)
        ktxx /= self.material.mu0
        kt1212 -= ktxx
        
        kt1212 *= self.getFactor()
        kt1212 *= (2.0-self.material.mur)
#        kt1212 *= -1.0
        
        K[0,0] -= kt1212[0,0]
        K[0,1] -= kt1212[0,1]
        K[1,0] -= kt1212[1,0]
        K[1,1] -= kt1212[1,1]
        
        self.KtS[inod][jnod][0,0] -= kt1212[0,0]
        self.KtS[inod][jnod][0,1] -= kt1212[0,1]
        self.KtS[inod][jnod][1,0] -= kt1212[1,0]
        self.KtS[inod][jnod][1,1] -= kt1212[1,1]
        
        # Mechanics - Magnetic coupling
        rotN = np.array([self.rotN[0,jnod],0.0,self.rotN[1,jnod]]) 
#        en = np.einsum('i,ij,j',rotN,self.Cg,condt)*(2.0-self.material.mur)
        en = np.einsum('i,ij,j',rotN,self.Cg,condt)
        en *= 1.0/(self.JF*self.material.mu0)
        kt123 = en*np.einsum('ijl,jl',self.gradTens[inod,:,:,:],self.Fmg)
        
        te = np.einsum('ijl,j,lk,k',self.gradTens[inod,:,:,:],rotN,self.Fg,\
                       condt)
        te += np.einsum('ijl,j,lk,k',self.gradTens[inod,:,:,:],condt,self.Fg,\
                       rotN)
        te *= 1.0/(self.JF*self.material.mu0)
        kt123 -= te
        kt123 *= self.getFactor()
        
        kt123 *= (2.0-self.material.mur)
#        kt123 *= -1.0
#        K[0,2] -= kt123[0]
#        K[1,2] -= kt123[1]
    
    
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
        condt = np.array([0.0,0.0,self.condt[1]])
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
        te = en*self.Fmg
        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
        te /= self.material.mu0
        te /= self.JF
        rie = np.einsum('kij,ij',self.gradTens[inod,:,:,:],te)
        rie *= self.getFactor()
                
        R[0] -= t*t*rie[0]
        R[1] -= t*t*rie[1]
        
        self.RiLambd[inod][0] -= t*rie[0]
        self.RiLambd[inod][1] -= t*rie[1]
        
        #Magneto-mechanics coupling
#        condt = np.array([self.rotA[0],0.0,self.rotA[1]])
##        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)*(2.0-self.material.mur)
#        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)
#        te = en*self.Fmg
#        te -= np.einsum('i,jk,k',condt,self.Fg,condt)
#        te /= self.material.mu0
#        te /= self.JF
#        rie = np.einsum('kij,ij',self.gradTens[inod,:,:,:],te)
#        rie *= self.getFactor()
#        
##        rie *= (2.0-self.material.mur)
##        rie *= -1.0
#        R[0] -= rie[0]
#        R[1] -= rie[1]
        
        #Magnetics part
        r3 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotA)
        r3 /= self.JF*self.material.mu0
#        r3 = np.dot(self.rotN[:,inod],self.rotA)
#        r3 /= self.material.mu0
#        R[2] += r3*self.getFactor()
        
#    def calculateRe(self, R, inod, t):
###        re = self.N_[inod]*self.getBodyLoad(t)
###        re *= -self.getFactor()*self.material.rho
###        R[0] = 0.0
###        R[1] = re
###        R[2] = 10.0*self.getFactor()
#        condt = self.condt
#        en = 0.5*np.dot(condt,condt)
##        en = 0.5*np.einsum('i,ij,j',condt,self.Cgx,condt)
#        te = en*np.eye(2,dtype=self.dtype)
#        te -= np.einsum('i,j',condt,condt)
#        te *= (2.0-self.material.mur)
#        te /= self.material.mu0
##        te -= np.einsum('i,jk,k',condt,self.Fgx,condt)
##        te /= self.material.mu0*self.JF
##        te = en*self.Fmg - np.einsum('i,j',condt,np.dot(self.Fg,condt))
###        te *= (2.0-self.material.mur)/self.material.mu0
###        te *= (self.material.mur-1.0)/self.material.mu0
##        te *= 1.0/self.material.mu0/self.JF
###        te *= (2.0-self.material.mur)
#        rie = np.dot(te,self.dN_[:,inod])
#        rie *= self.getFactor()
#        R[0] = rie[0]
#        R[1] = rie[1]


class AxiSymMagMech(AE.AxiSymNeumann,AxiMechElement):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, intData,\
                 normv, ide, condt):
        AE.AxiSymNeumann.__init__(self,Nodes,pd,basisFunction,\
                                            nodeOrder,intData,normv,ide)
        self.rotNs = []
        for i in range(self.intData.getNumberPoint()):
            self.rotNs.append(np.empty((self.Ndim,self.Nnod),self.dtype))
        self.rotA = np.empty(self.Ndim,self.dtype)
        self.RiLambd = []
        for i in range(self.Nnod):
            self.RiLambd.append(ia.zeros(self.Ndof,self.dtype))
        
        self.kt = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.ktT = np.empty((self.Ndof,self.Ndof),self.dtype)
        self.rie = np.empty(self.Ndof,self.dtype)
        self.tangent = np.cross(np.array([self.normv[0],self.normv[1],0.0]),\
                                         np.array([0.0,0.0,1.0]))[0:2]
        
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
        
        self.KtT = []
        self.KtS = []
        for i in range(self.Nnod):
            self.KtT.append([])
            self.KtS.append([])
            for j in range(self.Nnod):
                self.KtT[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
                self.KtS[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
        
        self.tangent /= np.linalg.norm(self.tangent)
        self.condt = condt
#        self.linear = True
    
    def getFactor(self):
        """
        Return: factor for integration, 2*pi*radius*det(J)
        """
        return 2*np.pi*self.x_[0]*self.factor[self.ig]
#        return self.factor[self.ig]
        
    def multiplyFmderiv(self, b, jnod):
        kt = np.einsum('kj,i,im,nj,lmn',self.ident1,b,self.Fmg,\
                       self.Fmg,self.gradTens[jnod,:,:,:])
        kt *= -1.0
        return kt
    
    def multiplyFderiv(self,a,b,c,jnod):
        nB = np.dot(a,b)
        kt = nB*np.einsum('kp,lpq,q',self.ident1,\
                        self.gradTens[jnod,:,:,:],c)
        kt /= self.JF
        return kt
     
#    def calculateKLinear(self,K,inod,jnod,t):
#        pass
    def calculateK(self, K, inod, jnod, t):
        
#        condt = t*np.array([0.0,0.0,self.condt[1]],dtype = self.dtype)
        condt = np.array([0.0,0.0,self.condt[1]],dtype = self.dtype)
#        condt = np.array([self.rotA[0],0.0,self.rotA[1]],dtype = self.dtype)
        normv = np.array([self.normv[0],0.0,self.normv[1]],dtype=self.dtype)
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)

        te = en*np.dot(self.normv,self.Fmgx)

#        te -= np.dot(normv,condt)*np.dot(self.Fgx,t*self.condt)
        te -= np.dot(normv,condt)*np.dot(self.Fgx,self.condt)
        kty = self.derivativeJinv(jnod)
        kt1212 = np.einsum('i,j',te,kty)
        
        ktx = 0.5*self.multiplyCderiv(condt,condt,jnod)
        kt1212 += np.einsum('i,ij,k',self.normv,np.eye(2,dtype=self.dtype),ktx)
        
        en = 0.5 * np.einsum('i,ij,j',condt,self.Cg,condt)/self.JF
        kty = en*self.multiplyFmderiv(normv,jnod)
        kt1212 += kty
        ktz = self.multiplyFderiv(normv,condt,condt,jnod)

        kt1212 -= ktz
        kt1212 /= self.material.mu00
        kt1212 *= self.getFactor()*self.N_[inod]
        
#        kt1212 *= -1.0
        
        K[0,0] += kt1212[0,0]
        K[0,1] += kt1212[0,1]
        K[1,0] += kt1212[1,0]
        K[1,1] += kt1212[1,1]
        
        self.KtS[inod][jnod][0,0] += kt1212[0,0]
        self.KtS[inod][jnod][0,1] += kt1212[0,1]
        self.KtS[inod][jnod][1,0] += kt1212[1,0]
        self.KtS[inod][jnod][1,1] += kt1212[1,1]
        
#        K[2,0] += kt321[0]
#        K[2,1] += kt321[1]
        
    
    def calculateR(self, R, inod, t):
        condt = np.array([0.0,0.0,self.condt[1]],dtype = self.dtype)
#        condt = np.array([self.rotA[0],0.0,self.rotA[1]],dtype = self.dtype)
        en = 0.5*np.einsum('i,ij,j',condt,self.Cg,condt)/self.JF
#        en = 0.5*np.dot(condt,condt)
#        en /= self.JF
        ri12 = en*np.dot(self.normv,self.Fmgx)
#        ri12 = en*np.dot(self.normv,np.eye(2,dtype=self.dtype))
        nt = np.dot(self.normv,self.condt)*np.dot(self.Fgx,self.condt)
#        nt = np.dot(self.normv,self.condt)*self.condt
        nt /= self.JF
        ri12 -= nt
        ri12 /= self.material.mu00
        ri12 *= self.N_[inod]*self.getFactor()
        
#        ri12 *= -1.0
        
        R[0] += t*t*ri12[0]
        R[1] += t*t*ri12[1]
        self.RiLambd[inod][0] += t*ri12[0]
        self.RiLambd[inod][1] += t*ri12[1]
#        ri12 *= -1.0
#        R[0] += ri12[0]
#        R[1] += ri12[1]
#        R[2] += ri3
        
#    def calculateRe(self, R, inod, t):
#        condt = np.array(self.condt)
#        en = 0.5*np.dot(condt,condt)
##        en = 0.5*np.einsum('i,ij,j',condt,self.Cgx,condt)
#        tem = en*np.eye(2,dtype=self.dtype)
##        tem = en*self.Fmgx
##        tem -= np.einsum('i,j',condt,np.dot(self.Fgx,condt))
#        tem -= np.einsum('i,j',condt,condt)
##        tem /= self.material.mu00*self.JF
#        tem /= self.material.mu00
#        ri12 = np.dot(self.normv,tem)
#        ri12 *= self.N_[inod]
#        ri12 *= self.getFactor()
###        
###        ri12 = np.dot(self.normv,tem)
###        ri12 *= self.getFactor()
###        condt = self.condt[1]
###        ri12 = np.array([0.0,-condt],self.dtype)
###        ri12 = condt
###        ri12 *= self.N_[inod]
###        ri12 *= self.getFactor()
##        
#        R[0] = -ri12[0]
#        R[1] = -ri12[1]        
        
Ndof = 2             
tOrder = 0
Ng = [3,3]
numberSteps = 40
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)

condt = np.array([0.0,1.0])
#condt = np.array([1.0e7,0.0])

def loadfunc(x, t):
    return 9.8

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh():
    H = 0.01
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
    polys[0].setDivisionEdge13(2)
    polys[0].setDivisionEdge24(1)
    
    mat1 = LinearMechanicMaterial(2.1e11, 0.3, 7.8e3,10000.0)
    
    polys[0].setMaterial(mat1)
    
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    
    for n in nodesx:
#        n.setConstraint(False, 0.0, 0)
#        n.setConstraint(False, 0.0, 1)
        if math.fabs(n.getX()[0]-R)<1.0e-14 and math.fabs(n.getX()[1]-H*0.5)<1.0e-14:
#        if math.fabs(n.getX()[0]-R)<1.0e-14 :
#            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
#            ncenter = n
#            n.setLoad(-1.0e6,0)
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
        if math.fabs(n.getX()[0]-R)<1.0e-14 :
            n.setConstraint(False, 0.0, 0)
#            n.setConstraint(False, 0.0, 1)
#            n.setConstraint(False, condt[1]*0.5*n.getX()[0],2)
            
        if math.fabs(n.getX()[0])<1.0e-14:
            n.setConstraint(False, 0.0, 0)
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
#        elements[-1].setBodyLoad(loadfunc)
    
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    elementBs = []
    condtx = condt
    for i,e in enumerate(elems1):
#        if np.fabs(e[1].getX()[0]-R) < 1.0e-13:
#            condtx = condt
#            elementBs.append(AxiSymMagMech(e,2,QE.LagrangeBasis1D,\
#            QE.generateQuadNodeOrder(2,1),intDatB,normBndVec[i],i,condtx))
#            elementBs[-1].setMaterial(mat1)
#        else:
#            condtx = np.array([0.0,0.0])
        if np.fabs(e[1].getX()[0]) < 1.0e-13:
            continue
        elementBs.append(AxiSymMagMech(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,normBndVec[i],i,condtx))
        elementBs[-1].setMaterial(mat1)
        if np.fabs(e[1].getX()[1])<1.0e-14:
            elementBs[-1].normv = np.array([0.0,-1.0])
        if np.fabs(e[1].getX()[1]-H)<1.0e-14:
            elementBs[-1].normv = np.array([0.0,1.0])
    

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
#alg = NR.LinearStabilityProblem(mesh,output,sv.numpySolver())
alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
                                          arcl=10.0)
#alg.enableVariableConstraint()
#alg.calculate(True)
#alg.calculate(False)
alg.calculate()

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)
        