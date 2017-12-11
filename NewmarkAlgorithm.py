# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 18:36:10 2017

@author: haiau
"""

import FEMAlgorithm as FA
import numpy as np

class GeneralizedAlphaAlgorithm(FA.DynamicAlgorithm):
    """
    Newmark algorithm
    """
    def __init__(self, Mesh, timeOrder, output, solver,\
    totalTime, numberTimeSteps, spectral_radius):
        """
        Initialize Newmark Algorithm Object
        Input:
            Mesh: mesh of problem, Mesh object
            timeOrder: degree of differential equation in time, integer
            output: output of problem, FEMOutput object
            solver: matrix solver, Solver object
            totalTime: the final time instance, float
            numberTimeSteps: number of time steps, integer
            spectral_radius: spectral radius of method, float 0<spectral_radius<=1
        """
        FA.DynamicAlgorithm.__init__(self,Mesh,\
        timeOrder,output,solver,totalTime,numberTimeSteps)
        self.rho_inf = spectral_radius
        self.alpha_m = 0
        self.alpha_f = 0
        self.beta = 0.5
        self.gamma = 0.5
        self.u_n = np.zeros(self.Neq,self.dtype)
        self.u_n1 = np.zeros(self.Neq,self.dtype)
        self.v_n = np.zeros(self.Neq,self.dtype)
        self.__tempM__ = np.zeros((self.Neq,self.Neq),self.dtype)
        self.__tempR__ = np.zeros(self.Neq,self.dtype)
        self.a_n = None
        self.point_load = False
        self.time_depend_load = False
        if timeOrder == 2:
            self.a_n = np.zeros(self.Neq,self.dtype)
        
    def calculateParameters(self):
        """
        calculate parameters for alpha methods
        """
        self.alpha_m = (2.0*self.rho_inf-1.0)/(self.rho_inf+1.0)
        self.alpha_f = self.rho_inf/(self.rho_inf + 1.0)
        self.beta = (1.0-self.alpha_m+self.alpha_f)/2.0
        self.beta *= self.beta
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        
    def check_point_load(self):
        """
        Check if there is point load
        """
        for n in self.mesh.getNodes():
            if n.hasPointLoad():
                self.point_load = True
                break
        for n in self.mesh.getNodes():
            if n.timeDependentLoad():
                self.time_depend_load = True
                break
    
    def generateExternalLoad(self):
        """
        generate external load
        """
        if not self.time_depend_load and self.istep > 0:
            return
        if self.point_load:
            for n in self.mesh.getNodes():
                n.getPointLoadToGlobal(self.Re, self.deltaT*self.istep)
    
    def initialConditions(self):
        """
        generate initial condition
        """
        for node in self.mesh.getNodes():
            node.assembleU(self.u_n)
            if self.timeOrder > 0:
                node.assembleV(self.v_n)
            if self.timeOrder == 2:
                node.assembleA(self.a_n)
    
    def NewmarkApproximation(self, vn1, an1):
        """
        calculate Newmark approximation
        """
        a = self.gamma/(self.beta*self.deltaT)
        b = (self.gamma - self.beta)/self.beta
        c = (self.gamma - 2*self.beta)*self.deltaT/(2.0*self.beta)
        d = 1.0/(self.beta*self.deltaT*self.deltaT)
        e = 1.0/(self.beta*self.deltaT)
        f = (1.0 - 2*self.beta)/(2.0*self.beta)

        if self.timeOrder > 0:
            np.copyto(vn1,self.u_n1)
            vn1 -= self.u_n
            vn1 *= a
            np.copyto(self.__tempR__,self.v_n)
            self.__tempR__ *= b
            vn1 -= self.__tempR__
        
        if self.timeOrder == 2 and not an1 is None:
            np.copyto(an1,self.u_n1)
            an1 -= self.u_n
            an1 *= d
            np.copyto(self.__tempR__,self.v_n)
            self.__tempR__ *= e
            an1 -= self.__tempR__

            np.copyto(self.__tempR__,self.a_n)
            self.__tempR__ *= c
            vn1 -= self.__tempR__

            np.copyto(self.__tempR__,self.a_n)
            self.__tempR__ *= f
            an1 -= self.__tempR__

        #return vn1,an1
        
    def midpointApproximation(self, vn1, an1):
        """
        midpoint approximation
        Notice: this method mutate vn1 and an1!
        """
        if self.timeOrder == 2 and not an1 is None:
            np.copyto(self.A, self.a_n)
            self.A -= an1
            self.A *= self.alpha_m
            self.A += an1

        if self.timeOrder > 0:
            np.copyto(self.V,self.v_n)
            self.V -= vn1
            self.V *= self.alpha_f
            self.V += vn1
        
        np.copyto(self.U,self.u_n)
        self.U -= self.u_n1
        self.U *= self.alpha_f
        self.U += self.u_n1
        
    def calculateKEffect(self):
        """
        effective stiffness matrix
        """
        self.Kt *= (1-self.alpha_f)
        a = (1-self.alpha_m)/(self.beta*self.deltaT*self.deltaT)
        if self.timeOrder == 2 and not self.M is None:
            np.copyto(self.__tempM__,self.M)
            self.__tempM__ *= a
            #self.Kt += a*self.M
            self.Kt += self.__tempM__
        if self.timeOrder > 0:
            b = self.gamma*(1-self.alpha_f)/(self.beta*self.deltaT)
            np.copyto(self.__tempM__,self.D) 
            self.__tempM__ *= b
            #self.Kt += b*self.D
            self.Kt += self.__tempM__
    
    def calculateREffect(self):
        """
        effective load vector
        """
        self.Ri *= -1.0
        self.Ri += self.Re
        #if self.timeOrder > 0:
        #    np.dot(self.D,self.V,self.__tempR__)
        #    self.Ri -= self.__tempR__
        #if self.timeOrder == 2:
        #    np.dot(self.M,self.A,self.__tempR__)
        #    self.Ri -= self.__tempR__


class LinearAlphaAlgorithm(GeneralizedAlphaAlgorithm):
    """
    Linear Newmark Algorithm
    """
    def calculateREffect(self):
        """
        effective load vector
        """
        a = self.gamma*(1.0-self.alpha_f)/(self.beta*self.deltaT)
        b = (self.gamma-self.gamma*self.alpha_f-self.beta)/(self.beta)
        c = (self.gamma-2.0*self.beta)*(1.0-self.alpha_f)
        c *= self.deltaT/(2.0*self.beta)
        d = (1.0-self.alpha_m)/(self.beta*self.deltaT*self.deltaT)
        e = (1.0-self.alpha_m)/(self.beta*self.deltaT)
        f = (1.0-self.alpha_m-2*self.beta)/(2.0*self.beta)
        
        v = a*self.u_n + b*self.v_n
        if self.timeOrder == 2:
            v += c*self.a_n
            a = d*self.u_n + e*self.v_n + f*self.a_n
        self.Ri = np.dot(self.Kt,self.alpha_f*self.u_n)
        self.Ri += np.dot(self.D,v)
        if self.timeOrder == 2:
            self.Ri += np.dot(self.M,a)
            
    def calculate(self):
        """
        start analysis
        """
        print("Start Analysis")
        self.calculateParameters()
        self.initialConditions()
        self.calculateMatrices()
        self.calculateKEffect()
        # loop over time steps
        for self.istep in range(self.numberSteps):
            print("Time step "+str(self.istep))
            self.generateExternalLoad()
            self.calculateREffect()
            # solve system
            self.u_n1 = self.solver.solve(self.Kt,self.Ri)
            np.copyto(self.U,self.u_n1)
            self.NewmarkApproximation(self.v_n,self.a_n)
            np.copyto(self.V,self.v_n)
            np.copyto(self.A,self.a_n)
            np.copyto(self.u_n,self.u_n1)
            self.output.outputData(self)
        print("Finished!")
        self.output.finishOutput()
        
class NonlinearAlphaAlgorithm(GeneralizedAlphaAlgorithm):
    """
    Nonlinear Newmark algorithm
    """
    def __init__(self, Mesh, timeOrder, output, solver,\
    totalTime, numberTimeSteps, spectrum_radius, maxiter = 200,\
    tol = 1.0e-6, toltype = 0):
        """
        Initialize Nonlinear Newmark Algorithm
        """
        GeneralizedAlphaAlgorithm.__init__(self,Mesh,timeOrder,output,solver,\
        totalTime,numberTimeSteps, spectrum_radius)
        self.Niter = maxiter
        self.tol = tol
        self.toltype = toltype
        self.eta = 100*tol
        
    def isConverged(self):
        """
        Check if solution converged
        Return True if converged, False otherwise
        """
        if self.toltype == 0:
            eta = np.linalg.norm(self.Ri)/np.linalg.norm(self.u_n1-self.u_n)
            self.eta = eta
            print('eta = '+str(eta))
        if (eta <= self.tol):
            return True
        return False
        
    def getMaxNumberIter(self):
        """
        get maximum number of iterations
        """
        return self.Niter
        
    def calculate(self):
        """
        start analysis
        """
        print("Start Analysis")
        self.connect()
        self.check_point_load()
        self.calculateParameters()
        self.initialConditions()
        np.copyto(self.u_n1,self.u_n)
        vn1 = np.empty(self.Neq)
        an1 = None
        if self.timeOrder == 2:
            an1 = np.empty(self.Neq)
        # loop over time steps
        self.calculateLinearMatrices()
        for self.istep in range(1,self.numberSteps+1):
            print("Time step "+str(self.istep))
            self.generateExternalLoad()
            self.calculateExternalBodyLoad()
            # loop over iterations
            for iiter in range(self.Niter):
                self.NewmarkApproximation(vn1,an1)
                self.midpointApproximation(vn1,an1)
                #self.updateValues()
                self.calculateMatrices()
                self.addLinearMatrices()
                self.calculateKEffect()
                self.calculateREffect()
                
                self.solver.isolve(self.Kt,self.Ri)
                self.u_n1 += self.Ri
                if self.isConverged():
                    break
            if self.eta <= self.tol:
                print("Converged with eta = " + str(self.eta))
            else:
                raise NotConverged
            self.NewmarkApproximation(vn1,an1)
            np.copyto(self.U,self.u_n1)
            np.copyto(self.u_n,self.u_n1)
            try:
                np.copyto(self.v_n,vn1)
                np.copyto(self.a_n,an1)
                np.copyto(self.V,vn1)
                np.copyto(self.A,an1)
            except TypeError:
                pass
            self.output.outputData(self)
        print("Finished!")
        self.output.finishOutput()


class LinearNewmarkAlgorithm(LinearAlphaAlgorithm):
    """
    Newmark Algorithm
    alpha_f = 0.0
    alpha_m = 0.0
    beta = 0.5
    gamma = 0.5
    """
    def calculateParameters(self):
        """
        calculate parameters for alpha methods
        """
        self.alpha_f = 0.0
        self.alpha_m = 0.0
        self.beta = 1.0/(self.rho_inf+1.0)/(self.rho_inf+1.0)
        self.gamma = (3.0-self.rho_inf)/(2.0*self.rho_inf+2.0)     
        
class NonlinearNewmarkAlgorithm(NonlinearAlphaAlgorithm):
    """
    Newmark Algorithm
    alpha_f = 0.0
    alpha_m = 0.0
    beta = 0.5
    gamma = 0.5
    """
    def calculateParameters(self):
        """
        calculate parameters for alpha methods
        """
        self.alpha_f = 0.0
        self.alpha_m = 0.0
        self.beta = 1.0/(self.rho_inf+1.0)/(self.rho_inf+1.0)
        self.gamma = (3.0-self.rho_inf)/(2.0*self.rho_inf+2.0)    
                
class NotConverged(Exception):
    """
    Exception for convergence
    """              
    pass