# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:55:59 2017

@author: haiau
"""

import numpy as np
import FEMMesh as FM

class Algorithm(object):
    """
    Algorithm class stores properties and methods for algorithms
    """
    def __init__(self, Mesh, timeOrder, output, solver, dtype = 'float64'):
        """
        Initialize algorithm
        Input:
            Mesh: mesh to be analysed
            timeOrder: order of differential equation in time
            output: the output to write data
            solver: matrix solver
        """
        self.dtype = dtype
        self.output = output
        self.solver = solver
        self.mesh = Mesh
        self.timeOrder = timeOrder
        for i in range(Mesh.getNnod()):
            if Mesh.getNodes()[i].getTimeOrder() != timeOrder:
                raise AlgorithmTimeOrderMismatch
        
        try:        
            self.mesh.generateID()
        except FM.EmptyMesh:
            pass
                
        self.Neq = self.mesh.getNeq()
        self.NeqD = self.mesh.getNeqD()
        # Matrices and vectors
        # Stiffness matrix
        self.Kt = np.zeros((self.Neq,self.Neq),dtype)
        self.Ktd = np.zeros((self.Neq,self.NeqD),dtype)
        self.KtL = np.zeros((self.Neq,self.Neq),dtype)
        self.KtLd = np.zeros((self.Neq,self.NeqD),dtype)
        # Internal load vector
        self.Ri = np.zeros(self.Neq,dtype)
        self.Rid = np.zeros(self.NeqD,dtype)
        self.RiL = np.zeros(self.Neq,dtype)
        self.RiLd = np.zeros(self.NeqD,dtype)
        self.__tempRi__ = np.zeros(self.Neq,dtype)
        # External load vector
        self.Re = np.zeros(self.Neq,dtype)
        # Displacement
        self.U = np.zeros(self.Neq,dtype)
        self.Ud = np.zeros(self.NeqD,dtype)
        # Mass matrix
        self.M = None
        self.Md = None
        self.ML = None
        self.MLd = None
        # Damping matrix
        self.D = None
        self.Dd = None
        self.DL = None
        self.DLd = None
        # Velocity and acceleration vectors
        self.V = None
        self.A = None
        # If this is a dynamic problem, initialize D or M matrix
        if timeOrder > 0:
            self.D = np.zeros((self.Neq,self.Neq),dtype)
            self.Dd = np.zeros((self.Neq,self.NeqD),dtype)
            self.DL = np.zeros((self.Neq,self.Neq),dtype)
            self.DLd = np.zeros((self.Neq,self.NeqD),dtype)
            self.V = np.zeros(self.Neq,dtype)
            if timeOrder == 2:
                self.M = np.zeros((self.Neq,self.Neq),dtype)
                self.Md = np.zeros((self.Neq,self.NeqD),dtype)
                self.ML = np.zeros((self.Neq,self.Neq),dtype)
                self.MLd = np.zeros((self.Neq,self.NeqD),dtype)
                self.A = np.zeros(self.Neq)
        # Nonhomogeneous Dirichlet boundary condition 
        self.NonhomogeneousDirichlet = False
        for node in self.mesh.getNodes():
            if node.hasNonHomogeneousDirichlet():
                self.NonhomogeneousDirichlet = True
                break
        
    def getMesh(self):
        """
        Return the mesh
        """
        return self.mesh
        
    def getTimeOrder(self):
        """
        return time order
        """
        return self.timeOrder
        
    def getRe(self):
        return self.Re
    
    def getRi(self):
        return self.Ri
        
    def getRid(self):
        return self.Rid
        
    def getRiL(self):
        return self.RiL
        
    def getRiLD(self):
        return self.RiLd
        
    def getKt(self):
        return self.Kt
        
    def getKtd(self):
        return self.Ktd
        
    def getD(self):
        return self.D
        
    def getDd(self):
        return self.Dd
        
    def getM(self):
        return self.M
        
    def getMd(self):
        return self.Md
        
    def getKtL(self):
        return self.KtL
        
    def getKtLd(self):
        return self.KtLd
        
    def getDL(self):
        return self.DL
        
    def getDLd(self):
        return self.DLd
        
    def getML(self):
        return self.ML
        
    def getMLd(self):
        return self.MLd
        
    def getU(self):
        return self.U
        
    def getV(self):
        return self.V
        
    def getA(self):
        return self.A
        
    def checkDirichletBC(self):
        """
        check if system has nonhomogeneous Dirichlet boundary conditions
        """
        return self.NonhomogeneousDirichlet
        
    def calculateExternalPointLoad(self):
        """
        Calculate external point load and 
        nonhomogeneous Dirichlet boundary conditions
        """
        for node in self.mesh.getNodes():
            node.addLoadTo(self.Re)
            if self.NonhomogeneousDirichlet:
                node.assembleGlobalDirichlet(self.Ud)
                
    def calculateExternalBodyLoad(self):
        for e in self.mesh.getElements():
            e.calculateBodyLoad(self)
        
    def updateValues(self):
        """
        update global displacement, velocity and acceleration to every nodes
        """
        for node in self.mesh.getNodes():
            node.updateU(self.U)
            node.updateV(self.V)
            node.updateA(self.A)
            
    def connect(self):
        """
        Connect global vectors and matrices to nodes so that the assembling 
        processes can be ignored
        """
        for n in self.mesh.getNodes():
            n.connect(self.U,self.V,self.A)
        
        try:
            for e in self.mesh.getElements():
                e.connect(self.Ri, self.Rid, self.Kt, self.Ktd,\
                self.D, self.Dd, self.M, self.Md)
        except AttributeError:
            pass
            
    def calculateMatrices(self):
        """
        calculate non linear parts of matrices required for calculation
        """
        self.Kt.fill(0.0)
        if self.timeOrder == 2:
            self.M.fill(0.0)
        if self.timeOrder > 0:
            self.D.fill(0.0)
        self.Ri.fill(0.0)
        for element in self.mesh:
            if not element.isLinear():
                element.calculate(self)
            
    def calculateLinearMatrices(self):
        """
        calculate linear parts of matrices required for calculation
        """
        self.KtL.fill(0.0)
        if self.timeOrder == 2:
            self.ML.fill(0.0)
        if self.timeOrder > 0:
            self.DL.fill(0.0)
        for element in self.mesh:
            element.calculate(self,linear=True)       
            
    def addLinearMatrices(self):
        
        self.Kt += self.KtL
        self.RiL.fill(0.0)
        np.dot(self.KtL,self.U,self.RiL)
        try:
            self.D += self.DL
            np.dot(self.D,self.V,self.__tempRi__)
            self.RiL += self.__tempRi__
        except:
            pass
        try:
            self.M += self.ML
            np.dot(self.M,self.A,self.__tempRi__)
            self.RiL += self.__tempRi__
        except:
            pass
        self.Ri += self.RiL
            
    def calculate(self):
        """
        start analysis
        """
        pass
        
class AlgorithmTimeOrderMismatch(Exception):
    """
    Exception in case of time order of node is different than time order of
    algorithm
    """
    pass

class StaticAlgorithm(Algorithm):
    """
    Static algorithm
    """            
    def __init__(self, Mesh, output, solver):
        Algorithm.__init__(self, Mesh, 0, output, solver)
        
class DynamicAlgorithm(Algorithm):
    """
    Dynamic algorithm
    """
    def __init__(self, Mesh, timeOrder, output, solver,\
    totalTime, numberTimeSteps):
        Algorithm.__init__(self, Mesh, timeOrder, output, solver)
        self.totalTime = totalTime
        self.numberSteps = numberTimeSteps
        self.deltaT = self.totalTime/self.numberSteps
        self.istep = 0
        
    def getCurrentStep(self):
        """
        Return current time step
        """
        return self.istep
        
    def getTime(self):
        """
        Return the current time (of the problem)
        """
        return self.istep*self.deltaT
        
class LinearStaticAlgorithm(StaticAlgorithm):
    """
    Linar static algorithm
    """        
    def calculate(self):
        """
        start analysis
        """
        # calculate matrices and vector
        for element in self.mesh:
            element.calculate(self)
            
        # calculate external point load
        self.calculateExternalPointLoad()
        
        # homogeneous Dirichlet Boundary Condition
        if self.checkDirichletBC():
            self.Re -= np.dot(self.Ktd,self.Ud)
            
        # solve system
        self.U = self.solver.solve(self.Kt, self.Re)
        self.updateValues()
        # write data to output
        self.output.outputData(self)
        self.output.finishOutput()
        