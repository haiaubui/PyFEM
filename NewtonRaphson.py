#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 12:00:27 2018

@author: haiau
"""

import FEMAlgorithm as FA
import numpy as np
import scipy as scp

class LoadControlledNewtonRaphson(FA.StaticAlgorithm):
    """
    Load controlled Newton-Raphson algorithm
    """
    def __init__(self, Mesh, output, solver, numberLoadSteps=10, max_iter=200,\
                 tol=1.0e-8,toltype = 0):
        """
        Initialize LoadControlledNewtonRaphson Algorithm Object
        Input:
            Mesh: mesh of problem, Mesh object
            output: output of problem, FEMOutput object
            solver: matrix solver, Solver object
            numberLoadSteps: number of time steps, integer, default is 10
            max_iter: maximum number of itteration, integer, default is 200
        """
        FA.StaticAlgorithm.__init__(self,Mesh,output,solver)
        self.lmd = 0.0
        self.NLSteps = numberLoadSteps
        self.deltal = 1.0/self.NLSteps
        self.Niter = max_iter
        self.u_n = np.zeros(self.Neq,self.dtype)
        self.tol = tol
        self.toltype = toltype
        self.istep = 0
        self.varconstr = False
        self.diag = None
        if self.NonhomogeneousDirichlet:
            self.RD = np.zeros(self.NeqD,self.dtype)
        
    def getLambda(self):
        """
        Return lambda of current load step
        """
        return self.lmd
    
    def getTime(self):
        """
        Return lambda of current load step
        """
        return self.lmd
    
    def enableVariableConstraint(self):
        """
        Update Dirichlet boundary condition at each load step
        """
        self.varconstr = True
        
    def isConverged(self, deltaU):
        """
        Check if solution converged
        Return True if converged, False otherwise
        """
        nrmdeltaU = np.linalg.norm(deltaU)
        if self.toltype == 0:
            eta = nrmdeltaU/np.linalg.norm(self.U-self.u_n)
            self.eta = eta
            print('eta = '+str(eta))
        if (eta <= self.tol):
            return True
        # deltaU is too small, the nonconvergence is due to floating point
        # error
        nrmU = np.linalg.norm(self.U)
        eta = nrmdeltaU/nrmU
        if (eta <= 1.0e-14) or (nrmU < 1.0e-14 and nrmdeltaU <= 1.0e-14):
            raise FloatingPointError
        return False
        
    def getMaxNumberIter(self):
        """
        get maximum number of iterations
        """
        return self.Niter
    
    def getNumberLoadSteps(self):
        """
        return number of load steps
        """
        return self.NLSteps
    
    def detectUnstability(self):
        """
        return True if system has just passed a stability point
        """
        try:
            
#            self.diag = self.solver.LDLdiag(0.5*(self.Kt+self.Kt.T))
#            self.diag = self.solver.LDLdiag(self.Kt)
#            self.eigfull = np.linalg.eigvals(self.Kt + self.Kt.T)
#            self.eig1,self.Ueig = scp.linalg.eig(self.KtT,-self.KtS)
#            print('min eig T '+str(np.min(np.abs(self.eig1))))
            self.eig = np.linalg.eigvals(self.Kt)
            print('min eig '+str(np.min(self.eig)))
            if (np.iscomplex(self.eig)).any():
                raise LimitPoint
            if (self.eig < 0.0).any():
                return True
#            return False
#            print('smallest eigval ' + str(np.min(self.eig)))
#            print('smallest eigval full ' + str(np.min(self.eigfull)))
#            print('min diag '+ str(np.min(self.solver.LDLdiag(KtTest))))
#            if (np.iscomplex(self.eig)).any():
#                return True
#            if (self.diag < 0.0).any():
#                raise LimitPoint
        except np.linalg.LinAlgError:
            return True
#        self.eigprev = self.eig
#        if self.diag is None:
#            self.diag,_ = np.linalg.eig(self.Kt)
#            return False
#        
#        self.prevdiag = self.diag
#        self.diag,_ = np.linalg.eig(self.Kt)
#        nneg = np.sum(self.prevdiag < 0.0)
#        nneg1 = np.sum(self.diag < 0.0)
##        if nneg != nneg1:
##            return True
#        if nneg1 > 0 or (np.fabs(self.diag) < 1.0e-5).any():
#            return True
        
#        [P,L,U] = self.solver.LUdecomp(self.Kt)
#        D = np.diag(U)
#        
#        if (D < 0.0).any():
#            return True
        
        return False
        
    
    def calculateStabilityPoints(self):
        """
        calculate the stability points
        """
        Utemp = np.empty(self.Neq, self.dtype)
        Udtemp = np.empty(self.NeqD,self.dtype)
        KtTemp = np.empty((self.Neq,self.Neq),self.dtype)
        KtTemp0 = np.empty((self.Neq,self.Neq),self.dtype)
        KtdTemp = np.empty(self.Ktd.shape,self.dtype)
        KtdTemp0 = np.empty(self.Ktd.shape,self.dtype)
        ReTemp = np.empty(self.Neq,self.dtype)
        RiTemp = np.empty(self.Neq,self.dtype)
        Gvec = np.empty(self.Neq, self.dtype)
        np.copyto(ReTemp,self.Re)
        np.copyto(RiTemp,self.Ri)
        np.copyto(Utemp,self.U)
        np.copyto(Udtemp,self.Ud)
        np.copyto(KtdTemp0,self.Ktd)
        np.copyto(KtTemp0,self.Kt)
        lambd = self.lmd
        self.lmd = self.lmd0
        Ret = np.empty(self.Neq,self.dtype)
        phi0 = 1.0/self.Neq * np.ones(self.Neq,self.dtype)
        phi00 = np.empty(self.Neq,self.dtype)
        
        m = 3
        for k in range(m):
            self.solver.isolve(self.Kt,phi0)
        np.copyto(phi00,phi0)
        epsil = np.max(np.fabs(phi0))*1.0e-3
        print('Start finding stability points')
        for iiter in range(self.Niter):
            if self.varconstr:
                np.copyto(self.Ud,self.RD)
                self.Ud *= self.lmd
            self.calculateMatrices()
            self.addLinearMatrices()
            print('det '+\
                  str(np.min(np.abs(np.linalg.eigvals(self.Kt)))))
            np.copyto(KtTemp, self.Kt)
            np.copyto(KtdTemp,self.Ktd)
            np.copyto(Ret,self.Re)
            
            if self.varconstr:
                Retx = np.dot(self.Ktd,self.RD)
                deltaUP = self.solver.solve(self.Kt, Ret-Retx-self.RiLambd)
            else:
                deltaUP = self.solver.solve(self.Kt, Ret-self.RiLambd)
            Ret *= self.lmd
            self.Ri -= Ret
            np.copyto(Gvec, self.Ri)
            self.Ri *= -1.0
            deltaUG = self.solver.solve(self.Kt,self.Ri)
            self.U += epsil*phi0
            self.calculateMatrices()
            self.addLinearMatrices()
            h1 =  (np.dot(self.Kt,deltaUP)-self.Re)
            if self.varconstr:
                h1 += np.dot(self.Ktd,self.RD)
                
#            epsil = np.max(np.fabs(phi0))*1.0e-9    
            h1 *= 1.0/epsil
            h2 = np.dot(KtTemp,phi0) + 1.0/epsil*(np.dot(self.Kt,deltaUG)+Gvec)
            deltaPhi1 = self.solver.solve(KtTemp,-h1)
            deltaPhi2 = self.solver.solve(KtTemp,-h2)
            nphi = np.linalg.norm(phi0)
            gradphi = 1.0/(nphi)*phi0
            nphi -= 1.0
            deltaL =-(np.dot(gradphi,deltaPhi2)+nphi)/np.dot(gradphi,deltaPhi1)
            deltaU = deltaL*deltaUP + deltaUG
            deltaP = deltaL*deltaPhi1 + deltaPhi2
            
            self.lmd += deltaL
            self.U -= epsil*phi0
            self.U += deltaU
            phi0 += deltaP
            
            etaL = np.fabs(deltaL)/np.fabs(lambd-self.deltal - self.lmd)
            etaP = np.linalg.norm(deltaP)/np.linalg.norm(phi0 - phi00)
            etaU = np.linalg.norm(deltaU)/np.linalg.norm(self.U - Utemp)
            etam = np.linalg.norm(deltaU)/np.linalg.norm(self.U)
            print('Iteration : '+str(iiter))
            print('etaL: ' + str(etaL) + ' lambda = ' + str(self.lmd))
            print('etaP: ' + str(etaP)+' |phi| = ' + str(np.linalg.norm(phi0)))
            print('etaU: ' + str(etaU))
            print('etam: ' + str(etam))
            
            if (etaL < self.tol and etaP < self.tol and etaU < self.tol) or \
            etam < 1.0e-14:
                break
        Ures = np.empty(self.Neq,self.dtype)
        np.copyto(Ures,self.U)
        np.copyto(self.Ud,Udtemp)
        np.copyto(self.U,Utemp)
        np.copyto(self.Kt,KtTemp0)
        np.copyto(self.Re,ReTemp)
        np.copyto(self.Ri,RiTemp)
        np.copyto(self.Ktd,KtdTemp0)
        lmdlim = self.lmd
        self.lmd = lambd
        
        if iiter < self.Niter-1:
            try:
                self.output.outputStability(lambd,Ures)
            except AttributeError:
                pass
            print("Found one stability point at lambda = " + str(lmdlim))
#            print('phi*R = ' + str(np.dot(phi0,self.Re)))
            if np.fabs(np.dot(phi0,self.Re)) > 1.0e2:
                print('Limit point')
            else:
                print('Bifurcation point')
            return lambd,Ures,phi0
        print('Could not find any stability point!')
        raise FA.NotConverged

   
    
    def calculate(self, find_stability=False):
        """
        start analysis
        """
        print("Start Analysis")
        self.prepareElements()
        self.connect()
        
        self.calculateExternalPointLoad()
        self.calculateExternalBodyLoad()
        
        self.calculateLinearMatrices()
        Ret = np.empty(self.Neq,self.dtype)
        if self.NonhomogeneousDirichlet:
            np.copyto(self.RD,self.Ud)
        if find_stability:
            Ucur = np.empty(self.U.shape,self.dtype)
        
        for self.istep in range(self.NLSteps):
            print("Load step "+str(self.istep))
            self.lmd += self.deltal
            self.calculateMatrices()
            self.addLinearMatrices()                
            if find_stability:
                if self.detectUnstability():
                    np.copyto(Ucur,self.U)
#                    np.copyto(self.U,self.u_n)
                    try:
                        lmds,u_stabil = self.calculateStabilityPoints()
                    except FA.NotConverged:
                        pass
                    np.copyto(self.U,Ucur)
                    
            np.copyto(Ret,self.Re)
            self.solver.isolve(self.Kt,Ret)
            self.U += Ret
            np.copyto(Ret,self.Re)
            Ret *= self.lmd
            
            if self.varconstr:
                self.Ud *= self.lmd
            
            
            np.copyto(self.u_n,self.U)
            for iiter in range(self.Niter):
                self.calculateMatrices()
                self.addLinearMatrices() 
                self.Ri -= Ret
                self.Ri *= -1.0
                self.solver.isolve(self.Kt,self.Ri)
                self.U += self.Ri
                
                try:
                    if self.isConverged(self.Ri):
                        break
                except FloatingPointError:
                    self.eta = 0.0
                    print("Increment is smaller than round off error")
                    break
                
            if self.varconstr:
                np.copyto(self.Ud,self.RD)
                
            if self.eta <= self.tol:
                print("Converged with eta = " + str(self.eta))
            else:
                raise FA.NotConverged
                
            self.outputData()
            
        print("Finished!")
        self.finishOutput()
        
class ArcLengthControlledNewtonRaphson(LoadControlledNewtonRaphson):
    """
    Arc-Lenght Controlled Newton Raphson algorithm
    """
    def __init__(self, Mesh, output, solver, numberLoadSteps=10, max_iter=200,\
                 tol=1.0e-8,toltype = 0, arcl = 100.0):
        LoadControlledNewtonRaphson.__init__(self,Mesh,output,solver,\
                                             numberLoadSteps,max_iter,\
                                             tol,toltype)
        self.arcl = arcl
        self.lmd = 0.0
        self.RiLambd = np.zeros(self.Neq,dtype=self.dtype)
        self.KtT = np.zeros((self.Neq,self.Neq),dtype=self.dtype)
        self.KtS = np.zeros((self.Neq,self.Neq),dtype=self.dtype)
        
    def connect(self):
        """
        Connect global vectors and matrices to nodes so that the assembling 
        processes can be ignored
        """
        for n in self.mesh.getNodes():
            n.connect(self.U,self.V,self.A,self.Ud)
        
        try:
            for e in self.mesh.getElements():
                e.connect(self.Ri, self.Rid, self.Kt, self.Ktd,\
                self.D, self.Dd, self.M, self.Md, self.RiLambd,\
                self.KtT,self.KtS)
        except AttributeError:
            pass    
        
    def calculateMatrices(self):
        """
        calculate non linear parts of matrices required for calculation
        """
        self.RiLambd.fill(0.0)
        FA.Algorithm.calculateMatrices(self)
        
    def getKtMechanicsOnly(self):
        dofs = []
        for n in self.mesh.Nodes:
            a = n.getID()
            if a[1] >= 0 and (a[1] not in dofs):
                dofs.append(a[1])
            if a[2] >= 0 and (a[2] not in dofs):
                dofs.append(a[2])
        dofs = np.array(dofs)
        KtTest = self.Kt[dofs[:,None],dofs]
        return KtTest

    
    def condFunc(self, un, u0, xn, x0):
        phi = 1.0
        s = self.arcl/self.NLSteps
        return np.sqrt(np.dot(un-u0,un-u0)+phi*phi*(xn-x0)*(xn-x0))-s
#        return un[0] + 2.0e-3*(self.istep + 1)
#        s = self.arcl/self.NLSteps
#        return np.linalg.norm(un - u0) - s
        
    def deltaLFunc(self, fc, uk, un, deltaur, deltaul, lamb0):
        phi = 1.0
        s = self.arcl/self.NLSteps
        return -(fc*(fc+s)+np.dot(uk-un,deltaur))/\
        (np.dot(uk-un,deltaul)+phi*phi*(self.lmd-lamb0))
#        return -(fc+deltaur[0])/deltaul[0]
#        return -(fc*(fc+s)+np.dot(uk-un,deltaur))/(np.dot(uk-un,deltaul))
    
    def calculateStabilityBisection(self):
        Utemp = np.empty(self.Neq, self.dtype)
        Udtemp = np.empty(self.NeqD,self.dtype)
        KtTemp = np.empty((self.Neq,self.Neq),self.dtype)
        KtTemp0 = np.empty((self.Neq,self.Neq),self.dtype)
        KtdTemp = np.empty(self.Ktd.shape,self.dtype)
        KtdTemp0 = np.empty(self.Ktd.shape,self.dtype)
        ReTemp = np.empty(self.Neq,self.dtype)
        RiTemp = np.empty(self.Neq,self.dtype)
        Gvec = np.empty(self.Neq, self.dtype)
        u0 = np.empty(self.Neq,self.dtype)
        np.copyto(ReTemp,self.Re)
        np.copyto(RiTemp,self.Ri)
        np.copyto(Utemp,self.U)
        np.copyto(Udtemp,self.Ud)
        np.copyto(KtdTemp0,self.Ktd)
        np.copyto(KtTemp0,self.Kt)
        lambd = self.lmd
        self.lmd = self.lmd0
        print(self.lmd0)
        Ret = np.empty(self.Neq,self.dtype)
        arcl = self.arcl
        
        for i in range(100):
            arcl_old = self.arcl
            self.arcl *= 0.5   
            print('arcl='+str(self.arcl))
            if self.arcl < 2.0e-5 :
                return self.calculateStabilityPoints()
            s = self.arcl/self.NLSteps
            lmd0 = self.lmd
            np.copyto(u0,self.U)
            if self.varconstr:
                np.copyto(self.Ud,self.RD)
            self.Ud *= self.lmd
            self.calculateMatrices()
            self.addLinearMatrices()
#            print('det='+str(np.min(np.fabs(np.linalg.eigvals(self.Kt))))) 
            np.copyto(Ret,self.Re)
            if self.varconstr:
                Ret -= np.dot(self.Ktd,self.RD)
            self.solver.isolve(self.Kt,Ret)
            s0 = np.sqrt(np.dot(Ret,Ret)+1.0)
            deltaL = s/s0
            self.U += deltaL*Ret
            self.lmd += deltaL
            
            if self.varconstr:
                np.copyto(self.Ud,self.RD)
                self.Ud *= self.lmd
            for iiter in range(self.Niter):
                self.calculateMatrices()
                self.addLinearMatrices()
                
                np.copyto(Ret,self.Re)
                Ret *= self.lmd
                self.Ri -= Ret
                self.Ri *= -1.0
                self.solver.isolve(self.Kt,self.Ri)#deltaur
                np.copyto(Ret,self.Re)
                if self.varconstr:
                    Ret -= np.dot(self.Ktd,self.RD)
                self.solver.isolve(self.Kt,Ret)#deltaul
                
                fc = self.condFunc(self.U,u0,self.lmd,lmd0)
                
                deltalambda =\
                self.deltaLFunc(fc,self.U,u0,self.Ri,Ret,lmd0)
                
                Ret *= deltalambda
                self.Ri += Ret
                self.U += self.Ri
#                if self.varconstr:
#                    self.Ud *= (self.lmd + deltalambda)/self.lmd
                self.lmd += deltalambda
                if self.varconstr:
                    np.copyto(self.Ud,self.RD)
                    self.Ud *= self.lmd
                
                
                etal = np.fabs(deltalambda)/np.fabs(self.lmd-lmd0)
                etas = np.linalg.norm(self.Ri)/np.linalg.norm(u0-self.U) 
                etam = np.linalg.norm(self.Ri)/np.linalg.norm(self.U)
                self.eta = etas
                if np.isnan(etas):
#                    etas = 1.0
                    break
                print('etau='+str(etas))
                print('deltal='+str(deltalambda))
                try:
                    if (etas < self.tol and etal < self.tol) or \
                    etam < 1.0e-13 or np.fabs(deltalambda) < 1.0e-14:
                        print('converged at lambda = ' + str(self.lmd))
#                        print('det='+str(np.min(np.linalg.eigvals(self.Kt))))
                        break
                except FloatingPointError:
                    self.eta = 0.0
                    print("Increment is smaller than round off error")
                    break
            etas = np.linalg.norm(u0 - self.U)/np.linalg.norm(Utemp)
            etals = np.fabs(lmd0 - self.lmd)/np.fabs(self.lmd0-self.lmd)
            print('etas='+str(etas))
            print('etals='+str(etals))
#            print('deltal='+str(np.fabs(lmd0-self.lmd)))
            eigval = np.linalg.eigvals(self.Kt)
            if (etas < self.tol*1.0e-2 and etals < self.tol*1.0e-2) or \
            np.fabs(np.min(eigval)) < 5.0e-2:
#            if np.fabs(np.min(np.linalg.eigvals(self.Kt))) < 1.0:     
                print("Found one stability point at lambda = " + str(self.lmd))
                w,v = np.linalg.eig(self.Kt)
                wm = np.argmin(w)
                print('phi*Re = '+str(np.dot(v[:,wm],self.Re)))
                if np.dot(v[:,wm],self.Re) < 1.0e-5:
                    print('Bifurcation point')
                break
            if self.eta < self.tol or etam<1.0e-14 or \
            np.fabs(deltalambda)<1.0e-14:
#                print('converged')
                if etam < 1.0e-13:
                    print('too small disp increment')
                if np.fabs(deltalambda)<1.0e-15:
                    print('too small lambda increment')
                print('det='+str(np.min(np.linalg.eigvals(self.Kt)))) 
                if np.isnan(etas) or self.detectUnstability():
#                    print('diag' + str(np.min(self.diag)))
                    print('still unstable, repeat with smaller arcl')
                    np.copyto(self.U,u0)
                    self.lmd = lmd0                    
                else:
                    print('stable solution')
                    self.arcl = arcl_old
                    continue
            else:
                print('not converged')
                np.copyto(self.U,u0)
                self.lmd = lmd0
                self.arcl = arcl_old
                self.arcl *= 1.5
        
        Ures = np.empty(self.Neq,self.dtype)
        np.copyto(Ures,self.U)
        np.copyto(self.Ud,Udtemp)
        np.copyto(self.U,Utemp)
        np.copyto(self.Kt,KtTemp0)
        np.copyto(self.Re,ReTemp)
        np.copyto(self.Ri,RiTemp)
        np.copyto(self.Ktd,KtdTemp0)
        self.lmd = lambd
        self.arcl = arcl
        
        return self.lmd,Ures
    
    def calculate(self, find_stability = True):
        """
        start analysis
        """
        print("Start Analysis")
        self.prepareElements()
        self.connect()
        
        self.U.fill(0.0)
        self.calculateExternalPointLoad()
        self.calculateExternalBodyLoad()
        
        self.calculateLinearMatrices()
        Ret = np.empty(self.Neq,self.dtype)
        if self.NonhomogeneousDirichlet:
            np.copyto(self.RD,self.Ud)
        if find_stability:
            Ucur = np.empty(self.U.shape,self.dtype)
            
        s = self.arcl/self.NLSteps    
        
        trigger = False
#        trigger = True
        lmd0 = 0.0
        self.calculateMatrices()
        self.addLinearMatrices()  
        self.uprev = np.empty(self.Neq,self.dtype)
        for self.istep in range(self.NLSteps):
            print("Load step "+str(self.istep))
            np.copyto(self.uprev,self.u_n)
            np.copyto(self.u_n,self.U)
            self.lmd0 = lmd0
            lmd0 = self.lmd
            
#            if trigger or self.detectUnstability():
#            if self.istep == 3:
#                trigger = 1
            if trigger:
                sign = -1.0
                print('change predictor')
                trigger = False
                if find_stability:
                    np.copyto(Ucur,self.U)
                    np.copyto(self.U,self.uprev)
                    try:
#                        print('lambda = ' + str(self.lmd))
#                        self.lmdst,self.u_stabil,self.phi_stabil = \
#                        self.calculateStabilityPoints()
                        lmds,u_stabil = self.calculateStabilityBisection()
                    except FA.NotConverged:
                        pass
                    np.copyto(self.U,Ucur)
            else:
                sign = 1.0

                    
            np.copyto(Ret,self.Re)
            Ret -= self.RiLambd
            if self.varconstr:
                Ret -= np.dot(self.Ktd,self.RD)
            self.solver.isolve(self.Kt,Ret)
            s0 = sign*np.sqrt(np.dot(Ret,Ret)+1)
            deltaL = s/s0
            self.U += deltaL*Ret
            self.lmd += deltaL
            
            if self.varconstr:
                np.copyto(self.Ud,self.RD)
                self.Ud *= self.lmd
                
                
            for iiter in range(self.Niter):
                self.calculateMatrices()
                self.addLinearMatrices()
#                print(self.detectUnstability())
                np.copyto(Ret,self.Re)
                Ret *= self.lmd
                self.Ri -= Ret
                self.Ri *= -1.0
                self.solver.isolve(self.Kt,self.Ri)#deltaur
                np.copyto(Ret,self.Re)
                Ret -= self.RiLambd
                if self.varconstr:
                    Ret -= np.dot(self.Ktd,self.RD)
                self.solver.isolve(self.Kt,Ret)#deltaul
                
                fc = self.condFunc(self.U,self.u_n,self.lmd,lmd0)
                
                deltalambda =\
                self.deltaLFunc(fc,self.U,self.u_n,self.Ri,Ret,lmd0)
                
                Ret *= deltalambda
                self.Ri += Ret
                self.U += self.Ri
                self.lmd += deltalambda
                if self.varconstr:
                    np.copyto(self.Ud,self.RD)
                    self.Ud *= self.lmd
                
                
                etal = np.fabs(deltalambda)/np.fabs(self.lmd-lmd0)
#                self.detectUnstability()
#                print('etal = '+str(etal))    
#                self.detectUnstability()
                try:
                    if self.isConverged(self.Ri) and etal < self.tol:
#                        self.calculateMatrices()
#                        self.addLinearMatrices()
                        
#                        if self.lmd > 0.04:
#                            trigger = True
                        if find_stability and self.detectUnstability():
                            trigger = True
#                        if np.linalg.norm(self.U - self.uprev) < 1.0e-10:
#                            print('repeat value')
#                            np.copyto(self.U,self.u_n)
#                            self.lmd = lmd0
#                            if self.varconstr:
#                                np.copyto(self.Ud,self.RD)
#                                self.Ud *= self.lmd
#                            self.calculateMatrices()
#                            self.addLinearMatrices()
#                            trigger = True
                        break
#                    if np.fabs(deltalambda)<1.0e-14:
#                        if self.detectUnstability():
#                            trigger = True
#                        break
                except FloatingPointError:
                    np.copyto(self.U,self.u_n)
                    self.calculateMatrices()
                    self.addLinearMatrices()
                    if find_stability and self.detectUnstability():
                        trigger = True
                    self.eta = 0.0
                    print("Increment is smaller than round off error")
                    break
                
            
            
            if not trigger:
#                self.detectUnstability()
                if self.eta <= self.tol or np.fabs(deltalambda)<1.0e-14:
                    print("Converged with eta = " + str(self.eta))
                    print('lambda = ' + str(self.lmd))
#                    print(self.mesh.Nodes[12].getU().tolist())
                else:
                    raise FA.NotConverged
                
            self.outputData()
            
        print("Finished!")
        self.finishOutput()

class LinearStabilityProblem(LoadControlledNewtonRaphson):
    """
    Algorithm to find stability point which appears at small or no displacement
    """
    def __init__(self, Mesh, output, solver):
        LoadControlledNewtonRaphson.__init__(self,Mesh,output,solver)
        
        self.KtT = np.zeros((self.Neq,self.Neq),dtype=self.dtype)
        self.KtS = np.zeros((self.Neq,self.Neq),dtype=self.dtype)
        
    def connect(self):
        """
        Connect global vectors and matrices to nodes so that the assembling 
        processes can be ignored
        """
        for n in self.mesh.getNodes():
            n.connect(self.U,self.V,self.A,self.Ud)
        
        try:
            for e in self.mesh.getElements():
                e.connect(self.Ri, self.Rid, self.Kt, self.Ktd,\
                self.D, self.Dd, self.M, self.Md, self.KtT, self.KtS)
        except AttributeError:
            pass    
    
    def calculateMatrices(self):
        """
        calculate non linear parts of matrices required for calculation
        """
        self.KtT.fill(0.0)
        self.KtS.fill(0.0)
        self.lmdeig = None
        self.Ueig = None
        FA.Algorithm.calculateMatrices(self)
    
    def calculate(self, find_stability = True):
        """
        start analysis
        """
        print("Start Analysis")
        self.prepareElements()
        self.connect()
        
        self.U.fill(0.0)
#        self.calculateExternalPointLoad()
#        self.calculateExternalBodyLoad()
        
        self.calculateLinearMatrices()
#        Ret = np.empty(self.Neq,self.dtype)
        if self.NonhomogeneousDirichlet:
            np.copyto(self.RD,self.Ud)
#        if find_stability:
#            Ucur = np.empty(self.U.shape,self.dtype)
            
#        s = self.arcl/self.NLSteps    
        
#        trigger = False
#        trigger = True
#        lmd0 = 0.0
        self.calculateMatrices()
        self.addLinearMatrices()  
        
#        matinv = np.linalg.solve(self.KtS,self.KtT)
        self.lmdeig,self.Ueig = scp.linalg.eig(self.KtT,-self.KtS)


class VariableControlledNewtonRaphson(LoadControlledNewtonRaphson):
    """
    Algorithm of variable controlled Newton Raphson method
    """
    def prepareControlVar(self):
        self.UdC = np.zeros(self.NeqD,dtype=self.dtype)
        self.controlList = [False]*self.NeqD
        for n in self.mesh.Nodes:
            if n.controlledDOF >= 0:
                self.controlList[-n.IDC-2]=True
                
        np.copyto(self.UdC,self.Ud,where=self.controlList)
        
    def mulControlVar(self):
        np.copyto(self.RD,self.UdC)
        self.RD *= self.lmd
        np.copyto(self.Ud,self.RD,where=self.controlList)
        
    def calculateStabilityBisection(self):
        Utemp = np.empty(self.Neq, self.dtype)
        Udtemp = np.empty(self.NeqD,self.dtype)
        KtTemp0 = np.empty((self.Neq,self.Neq),self.dtype)
        KtdTemp0 = np.empty(self.Ktd.shape,self.dtype)
        ReTemp = np.empty(self.Neq,self.dtype)
        RiTemp = np.empty(self.Neq,self.dtype)
        u0 = np.empty(self.Neq,self.dtype)
        np.copyto(ReTemp,self.Re)
        np.copyto(RiTemp,self.Ri)
        np.copyto(Utemp,self.U)
        np.copyto(Udtemp,self.Ud)
        np.copyto(KtdTemp0,self.Ktd)
        np.copyto(KtTemp0,self.Kt)
        lambd = self.lmd
        self.lmd = self.lmd0
        print(self.lmd0)
        Ret = np.empty(self.Neq,self.dtype)
        deltal = self.deltal
        
        for i in range(100):
            lmd0 = self.lmd
            np.copyto(u0,self.U)
            deltal_old = self.deltal
            self.deltal *= 0.5   
            self.lmd += self.deltal
            print('deltal='+str(self.deltal))
            print('lambda='+str(self.lmd))
            if self.deltal < 2.0e-8 :
                return self.calculateStabilityPoints()
            self.mulControlVar()
            for iiter in range(self.Niter):
                self.calculateMatrices()
                self.addLinearMatrices()
                
                np.copyto(Ret,self.Re)
                Ret *= self.lmd
                self.Ri -= Ret
                self.Ri *= -1.0
                self.solver.isolve(self.Kt,self.Ri)#deltaur
                
                self.U += self.Ri
                
                try:
                   if self.isConverged(self.Ri):
                        break
                except FloatingPointError:
                    self.eta = 0.0
                    print("Increment is smaller than round off error")
                    break
            etas = np.linalg.norm(u0 - self.U)/np.linalg.norm(Utemp)
            print('etas='+str(etas))
            eigval = np.linalg.eigvals(self.Kt)
            if etas < self.tol*1.0e-2 or np.fabs(np.min(eigval)) < 5.0e-2:
#            if np.fabs(np.min(np.linalg.eigvals(self.Kt))) < 1.0:     
                print("Found one stability point at lambda = " + str(self.lmd))
                w,v = np.linalg.eig(self.Kt)
                wm = np.argmin(w)
                print('phi*Re = '+str(np.dot(v[:,wm],self.Re)))
                if np.dot(v[:,wm],self.Re) < 1.0e-5:
                    print('Bifurcation point')
                break
            if self.eta < self.tol:
                print('det='+str(np.min(np.linalg.eigvals(self.Kt)))) 
                if np.isnan(etas) or self.detectUnstability():
#                    print('diag' + str(np.min(self.diag)))
                    print('still unstable, repeat with smaller arcl')
                    np.copyto(self.U,u0)
                    self.lmd = lmd0                    
                else:
                    print('stable solution')
                    self.deltal = deltal_old
                    continue
            else:
                print('not converged')
                np.copyto(self.U,u0)
                self.lmd = lmd0
                self.deltal = deltal_old
                self.deltal *= 2.0
        
        Ures = np.empty(self.Neq,self.dtype)
        np.copyto(Ures,self.U)
        np.copyto(self.Ud,Udtemp)
        np.copyto(self.U,Utemp)
        np.copyto(self.Kt,KtTemp0)
        np.copyto(self.Re,ReTemp)
        np.copyto(self.Ri,RiTemp)
        np.copyto(self.Ktd,KtdTemp0)
        self.lmd = lambd
        self.deltal = deltal
        
        return self.lmd,Ures
        
    def calculate(self, find_stability = True):
        """
        start analysis
        """
        print("Start Analysis")
        self.prepareElements()
        self.connect()
        self.prepareControlVar()
        
        self.U.fill(0.0)
        self.calculateExternalPointLoad()
        self.calculateExternalBodyLoad()
        
        self.calculateLinearMatrices()
        if self.NonhomogeneousDirichlet:
            np.copyto(self.RD,self.Ud)
        if find_stability:
            Ucur = np.empty(self.U.shape,self.dtype)
        
        trigger = False
#        trigger = True
        lmd0 = 0.0
        self.calculateMatrices()
        self.addLinearMatrices()  
        self.uprev = np.empty(self.Neq,self.dtype)
        for self.istep in range(self.NLSteps):
            print("Load step "+str(self.istep))
            np.copyto(self.uprev,self.u_n)
            np.copyto(self.u_n,self.U)
            
            self.lmd0 = lmd0
            lmd0 = self.lmd
            
#            if trigger or self.detectUnstability():
#            if self.istep == 3:
#                trigger = 1
            if trigger:
                print('instable region')
                trigger = False
                if find_stability:
                    np.copyto(Ucur,self.U)
                    np.copyto(self.U,self.uprev)
                    try:
#                        print('lambda = ' + str(self.lmd))
#                        self.lmdst,self.u_stabil,self.phi_stabil = \
#                        self.calculateStabilityPoints()
                        self.lmds,self.u_stabil = \
                        self.calculateStabilityBisection()
                    except FA.NotConverged:
                        pass
                    np.copyto(self.U,Ucur)
            
            self.lmd += self.deltal
            np.copyto(self.u_n,self.U)
            self.mulControlVar()    
                
            for iiter in range(self.Niter):
                self.calculateMatrices()
                self.addLinearMatrices()
                self.Ri -= self.Re
                self.Ri *= -1.0
                self.solver.isolve(self.Kt,self.Ri)
                self.U += self.Ri
                
                try:
                    if self.isConverged(self.Ri):
                        if self.detectUnstability():
                            trigger = True
                        break
                except FloatingPointError:
                    self.eta = 0.0
                    print("Increment is smaller than round off error")
                    break
                
            
            
            if not trigger:
                self.detectUnstability()
                if self.eta <= self.tol:
                    print("Converged with eta = " + str(self.eta))
                    print('lambda = ' + str(self.lmd))
#                    print(self.mesh.Nodes[12].getU().tolist())
                else:
                    raise FA.NotConverged
                
            self.outputData()
            
        print("Finished!")
        self.finishOutput()

class LimitPoint(Exception):
    """
    Exception if Limit Point happen at current point
    """

        