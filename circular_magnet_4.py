#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:41:16 2018

@author: haiau
"""

import math
import numpy as np
import sympy as syp
import scipy.special as scpsp
import pylab as pl
import AxisymmetricElement as AE
import FEMBoundary as FB
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
        en = 0.5*np.einsum('i,ij,j',rotN,self.Cg,condt)
        en += 0.5*np.einsum('i,ij,j',condt,self.Cg,rotN)
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
        k1 *= factor*r
        k1 += self.u_[idofA]*self.G*self.normv[0]*factor/r
        k2 = self.u_[idofJ]*self.G
        k2 *= factor*r
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
        
Ndof = 4             
tOrder = 0
Ng = [3,3]
numberSteps = 10
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(6, 1, idat.Gaussian1D)
intSingDat = SI.SingularGaussian1D(48, intDatB.xg,\
SI.Gaussian_1D_Pn_Log, SI.Gaussian_1D_Pn_Log_Rat)


condt = 50.0*np.array([np.cos(np.pi/2.0),\
                  np.sin(np.pi/2.0)])
#condt = np.array([1.0e5,0.0])

def loadfunc(x, t):
    return np.array([0.0,0.0,1000.0/0.01/0.01,0.0])

def loadfuncG(x, t):
    return np.array([0.0,-9.8*7.8e3,0.0,0.0])

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh():
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
        
#    nodes1 = []
#    nodes1.append(FN.Node([0,H*21],Ndof,timeOrder = tOrder))
#    nodes1.append(FN.Node([R*2,H*21],Ndof,timeOrder = tOrder))
#    nodes1.append(FN.Node([R*2+0.1*R,H*21],Ndof,timeOrder = tOrder))
#    
#    edges1 = [mg.Edge(nodes1[i],nodes1[i+1]) for i in range(len(nodes1)-1)]
#    
#    for e in edges1:
#        geo.addPolygons(e.extendToQuad(d,s))
        
#    nodes2 = []
#    nodes2.append(FN.Node([0,0],Ndof,timeOrder = tOrder))
#    nodes2.append(FN.Node([R,0],Ndof,timeOrder = tOrder))
#    
#    edges2 = [mg.Edge(nodes2[i],nodes2[i+1]) for i in range(len(nodes2)-1)]
#    
#    s = H
#    
#    for e in edges2:
#        geo.addPolygons(e.extendToQuad(d,s))
        
    polys = geo.getPolygons()
    
    for i in range(5):  
        polys[i].setDivisionEdge13(2)
        polys[i].setDivisionEdge24(1)
    for i in range(5,10):
        polys[i].setDivisionEdge13(2)
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
    mat2 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,1)
    mat3 = LinearMechanicMaterial(0.0, 0.0, 1.0,100.0,2)
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
        
    return mesh




mesh = create_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result_FE_2int.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg = NR.LinearStabilityProblem(mesh,output,sv.numpySolver())
alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
                                          arcl=10.0,max_iter=200)
#alg = NR.VariableControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg.enableVariableConstraint()
#alg.calculate(True)
alg.calculate(False)
#alg.calculate()

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)

        
