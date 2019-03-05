#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 17:17:09 2018

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
        
#        K[0,0] -= kt1212[0,0]
#        K[0,1] -= kt1212[0,1]
#        K[1,0] -= kt1212[1,0]
#        K[1,1] -= kt1212[1,1]
        
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
        
#        K[2,0] += kt321[0]
#        K[2,1] += kt321[1]
        
        
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
                
#        R[0] -= rie[0]
#        R[1] -= rie[1]
        
#        r3 = np.einsum('i,ij,j',self.rotN[:,inod],self.Cgx,self.rotA)
        r3 = np.einsum('i,i',self.rotN[:,inod],self.rotA)
#        r3 /= self.JF*self.material.mu0
        r3 /= self.material.mu0
        R[2] += r3*self.getFactor()
        
    def calculateRe(self, R, inod, t):
        R.fill(0.0)
        re = self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R[2] = re
        
        
class AxiSymMagneticBoundaryLinear(AE.AxisymmetricStaticBoundary,FB.StraightBoundary1D):
    def calculateGreen(self, x, xp):
        if np.allclose(x,xp,rtol=1.0e-13) or math.fabs(xp[0])<1.0e-14 or\
        math.fabs(x[0])<1.0e-14:
            self.G = 0.0
            try:
                self.gradG.fill(0.0)
                self.grgrG.fill(0.0)
            except AttributeError:
                pass
            raise FB.SingularPoint
        r = x[0]
        rp = xp[0]
        z = x[1]
        zp = xp[1]
        self.G = self.Gfunc(r,z,rp,zp)
        if np.isnan(self.G):
            print('nan here')
        self.gradG[0] = self.Gdr(rp,zp,r,z)
        self.gradG[1] = self.Gdz(rp,zp,r,z)
        self.gradG0[0] = self.Gdr0(rp,zp,r,z)
        self.gradG0[1] = self.Gdz0(rp,zp,r,z)
        self.grgrG[0,0] = self.Gdrr(rp,zp,r,z)
        self.grgrG[0,1] = self.Gdrz(rp,zp,r,z)
        self.grgrG[1,0] = self.Gdzr(rp,zp,r,z)
        self.grgrG[1,1] = self.Gdzz(rp,zp,r,z)
        self.gr0grG[0,0] = self.Gdr0r(rp,zp,r,z)
        self.gr0grG[0,1] = self.Gdr0z(rp,zp,r,z)
        self.gr0grG[1,0] = self.Gdz0r(rp,zp,r,z)
        self.gr0grG[1,1] = self.Gdz0z(rp,zp,r,z)
        if np.isnan(self.gr0grG).any():
            print('nan here')
            
    def postCalculateF(self, N_, dN_, factor, res):
        idofA = 0
        idofJ = 1
        r = self.x_[0]
        k1 = self.u_[idofA]*\
        (self.normv[0]*self.gradG[0]+self.normv[1]*self.gradG[1])
        k1 *= factor*r
#        k1 += self.u_[idofA]*self.G*self.normv[0]*factor/r
        k2 = self.u_[idofJ]*self.G
        k2 *= factor*r
        res += k1 - k2        
        
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
        K[idofA,idofA]=-self.N_[i]*element.Nx_[j]
        K[idofA,idofA]*=\
        np.einsum('i,ij,j',self.normv,self.gr0grG,element.normv)
        K[idofA,idofA]*= wfac*wfacx*r
#        K[idofA,idofA] = 0.0
        
        K[idofA,idofJ]=self.N_[i]*element.Nx_[j]
        K[idofA,idofJ]*=np.dot(self.normv,self.gradG0)
        K[idofA,idofJ]*= wfac*wfacx*r
#        K[idofA,idofJ] = 0.0
        
        K[idofJ,idofA]=self.N_[i]*element.Nx_[j]
        K[idofJ,idofA]*=np.dot(element.normv,self.gradG)
        
#        K[idofJ,idofA] -= self.N_[i]*element.Nx_[j]*self.G*element.normv[0]/r
        K[idofJ,idofA]*= wfac*wfacx*r
#        K[idofA,idofJ]=K[idofJ,idofA]
#        K[idofJ,idofA]=0.0
        
        K[idofJ,idofJ]=-self.N_[i]*element.Nx_[j]*self.G
        K[idofJ,idofJ]*= wfac*wfact*r
#        K[idofJ,idofJ] = 0.0
        
        K /= self.material.mu0
        K *= 2.0*np.pi
    
    def calculateKLinear(self, K, i, j, t):
#        K[2,3] = self.N_[i]*self.N_[j]
#        K[2,3] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
#        K[3,2] = -self.N_[i]*self.N_[j]*0.5
#        K[3,2] *= self.getFactor()/self.material.mu00
        K[0,1] = -0.5*self.N_[i]*self.N_[j]
        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
#        K[0,1] = 0.0 
        K[1,0] = -0.5*self.N_[i]*self.N_[j]
        K[1,0] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu00
        K[0,0] = -self.N_[i]*self.N_[j]*self.normv[0]
        K[0,0] *= self.getFactor()*2.0*np.pi/self.material.mu00
        K[1,1] = 0.0
#        K[0,0] = 0.0
        
#        K /= self.material.mu00

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
        self.tangent = np.cross(np.array(np.array([0.0,0.0,1.0])),\
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
                    
                    
    def postCalculateF(self, N_, dN_, factor, res):
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
        self.graduMx_.fill(0.0)
        for i in range(self.Nnod):
            np.einsum('i,ijk',self.Nodes[i].getU().toNumpy()[0:2],\
                      self.gradTensX[i,:,:,:],out=self.temp_graduM_)
            self.graduMx_ += self.temp_graduM_
        
    
    def calculateGreen(self, x, xp):
        if np.allclose(x,xp,rtol=1.0e-13) or math.fabs(xp[0])<1.0e-14 or\
        math.fabs(x[0])<1.0e-14:
            self.G = 0.0
            self.gradG.fill(0.0)
            self.grgrG.fill(0.0)
            raise FB.SingularPoint
        r = x[0]
        rp = xp[0]
        z = x[1]
        zp = xp[1]
        self.G = self.Gfunc(rp,zp,r,z)
        if np.isnan(self.G):
            print('nan here')
        self.gradG[0] = self.Gdr(rp,zp,r,z)
        self.gradG[1] = self.Gdz(rp,zp,r,z)
        self.gradG0[0] = self.Gdr0(rp,zp,r,z)
        self.gradG0[1] = self.Gdz0(rp,zp,r,z)
        self.grgrG[0,0] = self.Gdrr(rp,zp,r,z)
        self.grgrG[0,1] = self.Gdrz(rp,zp,r,z)
        self.grgrG[1,0] = self.Gdzr(rp,zp,r,z)
        self.grgrG[1,1] = self.Gdzz(rp,zp,r,z)
        self.gr0grG[0,0] = self.Gdr0r(rp,zp,r,z)
        self.gr0grG[0,1] = self.Gdr0z(rp,zp,r,z)
        self.gr0grG[1,0] = self.Gdz0r(rp,zp,r,z)
        self.gr0grG[1,1] = self.Gdz0z(rp,zp,r,z)
        
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
    
    def calculateKLinear(self, K, i, j, t):
        K[2,3] = self.N_[i]*self.N_[j]
        K[2,3] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
        
    def subCalculateKLinear(self, K, element, i, j):
        pass
        
    def calculateK(self, K, inod, jnod, t):
        r = self.x_[0]
        k30 = 0.5*self.N_[inod]*self.derivativeFm11(jnod)
        k30 *= self.getFactor()*r*2.0*np.pi
        K[3,0] -= k30[0]
        K[3,1] -= k30[1]
        k32 = 0.5*self.N_[inod]*self.N_[jnod]*self.Fmg[1,1]*self.getFactor()
        k32 *= r*2.0*np.pi
        K[3,2] -= k32
        
    def calculateR(self, R, inod, t):
        factor = self.getFactor()*2.0*np.pi*self.x_[0]
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
        
        k32 = -self.gradG[0]*element.normv[1]*Fzr
        k32 -= self.gradG[1]*element.normv[0]*Frz
        k32 += self.gradG[1]*element.normv[1]*Frr
        k32 += self.gradG[0]*element.normv[0]*Fzz
        
        k32 *= wfacx*wfac*self.N_[inod]*element.Nx_[jnod]
        
        k32 -= (element.normv[1]*Fzr-element.normv[0]*Fzz)/r*self.G*\
        self.N_[inod]*element.Nx_[jnod]*wfact*wfac
        
        K[3,2] = k32
        
        k3y = element.ux_[3]*dFpp_ur*self.G
        k3y *= self.N_[inod]*wfact*wfac
        K[3,0] = k3y
        
        k33 = element.Nx_[jnod]*Fpp*self.G
        k33 *= self.N_[inod]*wfact*wfac
        K[3,3] = k33
        
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
        ri -= self.G/r*(element.normv[1]*Fzr-element.normv[0]*Fzz)
        ri *= element.ux_[2]
        ri *= wfacx
        
        ri += element.ux_[3]*Fpp*self.G*wfact
        
        ri *= self.N_[inod]*wfac
        
        R[3] += ri
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None,\
             deformed = False, deformed_factor=1.0 ):
        return FE.StandardElement.plot(self,fig,col,fill_mat,number,\
                                deformed,deformed_factor)

class AxiSymMagnetic(AE.AxisymmetricQuadElement, QE.Quad9Element):
    def __init__(self, Nodes, pd, basisFunction, nodeOrder,material, intData):
        AE.AxisymmetricQuadElement.__init__(self,Nodes,pd,basisFunction,\
        nodeOrder,material,intData)
        self.store = True
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


Ndof = 4             
tOrder = 0
Ng = [3,3]
numberSteps = 2
tol = 1.0e-8

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)
intDatB = idat.GaussianQuadrature(6, 1, idat.Gaussian1D)
intSingDat = SI.SingularGaussian1D(24, intDatB.xg,\
SI.Gaussian_1D_Pn_Log_Rat, SI.Gaussian_1D_Pn_Log_Rat_Rat2)
#intSingDat = idat.GaussianQuadrature(12, 1, idat.Gaussian1D)

condt = np.array([np.cos(np.pi/2.0+1.0e-2*np.pi),\
                  np.sin(np.pi/2.0+1.0e-2*np.pi)])
#condt = np.array([1.0e5,0.0])

def loadfunc(x, t):
    return 1.0
nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

def create_mesh():
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
        
    nodes1 = []
    nodes1.append(FN.Node([0,H*21],Ndof,timeOrder = tOrder))
    nodes1.append(FN.Node([R*2,H*21],Ndof,timeOrder = tOrder))
    nodes1.append(FN.Node([R*2+0.1*R,H*21],Ndof,timeOrder = tOrder))
    
    edges1 = [mg.Edge(nodes1[i],nodes1[i+1]) for i in range(len(nodes1)-1)]
    
    for e in edges1:
        geo.addPolygons(e.extendToQuad(d,s))
        
    nodes2 = []
    nodes2.append(FN.Node([0,0],Ndof,timeOrder = tOrder))
    nodes2.append(FN.Node([R,0],Ndof,timeOrder = tOrder))
    
    edges2 = [mg.Edge(nodes2[i],nodes2[i+1]) for i in range(len(nodes2)-1)]
    
    s = H
    
#    for e in edges2:
#        geo.addPolygons(e.extendToQuad(d,s))
        
    polys = geo.getPolygons()
    
    polys[0].setDivisionEdge13(4)
    polys[0].setDivisionEdge24(1)
    polys[2].setDivisionEdge13(4)
    polys[2].setDivisionEdge24(1)
    
#    polys[4].setDivisionEdge13(8)
#    polys[4].setDivisionEdge24(1)
    
    mat1 = LinearMechanicMaterial(2.1e11, 0.3, 7.8e3,100.0,0)
    mat2 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,1)
    mat3 = LinearMechanicMaterial(0.0, 0.0, 1.0,100.0,2)
    mat4 = LinearMechanicMaterial(0.0, 0.0, 1.0,1.0,3)
    
    polys[0].setMaterial(mat3)
    polys[1].setMaterial(mat2)
    polys[2].setMaterial(mat3)
    polys[3].setMaterial(mat2)
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
        if m.idx == 1:
            elements[-1].setBodyLoad(loadfunc)
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
    
    mat1 = LinearMagneticMaterial(1.0,1.0,5.0e6,1)
    mat2 = LinearMagneticMaterial(1.0,1.0,5.0e6,2)
    poly1.setMaterial(mat1)
    poly2.setMaterial(mat2)
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
    elements = []
    
    load = 355.0/0.015/0.01
    
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

mesh = create_mesh()
#mesh = create_simple_mesh()

mesh.generateID()

output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')

#alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps)
#alg = NR.LinearStabilityProblem(mesh,output,sv.numpySolver())
alg = NR.ArcLengthControlledNewtonRaphson(mesh,output,sv.numpySolver(),numberSteps,\
                                          arcl=10.0,max_iter=200)
#alg.enableVariableConstraint()
#alg.calculate(True)
alg.calculate(False)
#alg.calculate()

#_,inod = mesh.findNodeNear(np.array([0.05,-0.005]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(10)),inod,'u')
#testout = [t[0][1] for t in testout]

#output.updateToMesh(mesh,9)
#X,Y,Z = mesh.meshgridValue([0.0,1.0,-0.05,0.05],0.01,1.0e-8)
        