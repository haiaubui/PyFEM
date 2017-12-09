# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:35:33 2017

@author: haiau
"""
import math
import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3
import FEMElement as FE
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewmarkAlgorithm as NM
import Solver as sv

# Material
class TrussMaterial(mat.Material):
    def __init__(self, E, A, rho):
        self.E = E
        self.A = A
        self.rho = rho
    
    def getE(self):
        return self.E
        
    def getA(self):
        return self.A
        
    def getEA(self):
        return self.E*self.A
        
    def getRhoA(self):
        return self.rho*self.A

# Crisfield element:
class CrisfieldElement(FE.Element):
    def __init__(self,Nodes, timeOrder, material, ndim):
        FE.Element.__init__(self,Nodes,0,None,material,None)
        self.L = np.linalg.norm(self.Nodes[1].getX()-self.Nodes[0].getX())
        self.Ndim = ndim
        
    def calculate(self, data, linear = True):
        Kt = data.getKt()
        Ktd = data.getKtd()
        M = data.getM()
        Md = data.getMd()
        
        u = (self.Nodes[1].getX()-self.Nodes[0].getX())
        a = self.material.getEA()/((self.L)**3)
        k = np.outer(u,u)
        k *= a
        m = (self.material.getRhoA()*self.L/12)*np.identity(self.Ndof)
        for i in range(self.Nnod):
            for j in range(self.Nnod):                
                K = (-1)**(i+j)*k
                m1 = ((-1)**(i+j)+3)*m
                FE.assembleMatrix(Kt,Ktd,K,self.Nodes[i],self.Nodes[j])
                FE.assembleMatrix(M,Md,m1,self.Nodes[i],self.Nodes[j])
                
# Nonlinear Crisfiedl Element:
class NonlinearCrisfieldElement(CrisfieldElement):
    def calculate(self, data, linear = False):
        if linear:
            return
        Kt = data.getKt()
        Ktd = data.getKtd()
        M = data.getM()
        Md = data.getMd()
        Ri = data.getRi()
        Rid = data.getRid()
        
        
        x = (self.Nodes[1].getX()-self.Nodes[0].getX())
        u = (self.Nodes[1].getU()-\
        self.Nodes[0].getU())
        b = self.material.getE()/(2.0*self.L*self.L)
        s11 = b*np.dot(-2*x-u,-u)
        re = self.material.getA()/self.L*s11*(x+u)
        a = self.material.getEA()/((self.L)**3)        
        km = np.outer(x+u,x+u)
        kg = self.material.getA()/self.L*s11*np.identity(self.Ndof)
        km *= a
        k = kg + km
        m = (self.material.getRhoA()*self.L/12)*np.identity(self.Ndof)
        for i in range(self.Nnod):
            R = (-1)**(i+1)*re
            FE.assembleVector(Ri,Rid,R,self.Nodes[i])
            for j in range(self.Nnod):                
                K = (-1)**(i+j)*k
                m1 = ((-1)**(i+j)+3)*m
                FE.assembleMatrix(Kt,Ktd,K,self.Nodes[i],self.Nodes[j])
                FE.assembleMatrix(M,Md,m1,self.Nodes[i],self.Nodes[j])

# mesh
class TrussMesh(FM.Mesh):
    def __init__(self, Ndim):
        FM.Mesh.__init__(self)
        self.Ndim = Ndim
        self.fig = None
        
    def updateValues(self, uGlob, vGlob, aGlob):
        for node in self.Nodes:
            node.updateU(uGlob)
            node.updateV(vGlob)
            node.updateA(aGlob)
    
    def plot(self, init = True, col='b'):
        """
        plot mesh
        This is an example to plot a truss structure, i.e. Ndime = 1
        The other plot function can be made by overide this method
        Input:
            col: color spec
        """
        if self.fig is None:
            self.fig = pl.figure()
        if self.Ndim == 3:
            ax = p3.Axes3D(self.fig)
        #loop over elements
        for i in range(self.Ne):
            e = self.Elements[i]
            #loop over dimensions
            if e.getNdim() == 1:
                for inod in range(e.getNnod()-1):
                    if init:
                        X = e.getNodes()[inod].getX()
                        X1 = e.getNodes()[inod+1].getX()
                    else:
                        X = e.getNodes()[inod].getX() + e.getNodes()[inod].getU()
                        X1 = e.getNodes()[inod+1].getX() +e.getNodes()[inod+1].getU()
                    xc = np.array([X[0],X1[0]])
                    if self.Ndim == 2:
                        yc = np.array([X[1],X1[1]])
                        pl.plot(xc,yc,col)
                    if self.Ndim == 3:
                        yc = np.array([X[1],X1[1]])
                        zc = np.array([X[2],X1[2]])
                        ax.plot(xc,yc,zc,col)
        return self.fig

# Build structure
def build_structure(E,rho,A,H,B1,B2,alpha,beta,timeOrder):
    node1 = FN.Node([-B2,H],2,timeOrder)
    node1.setConstraint(False, 0.0, 0)
    node1.setConstraint(False, 0.0, 1)
    node2 = FN.Node([0.0,0.0],2,timeOrder)
    node2.setConstraint(True, 0.0, 0)
    node2.setConstraint(False, 0.0, 1)
    node3 = FN.Node([B1,H],2,timeOrder)
    node3.setConstraint(False, 0.0, 0)
    node3.setConstraint(True, 0.0, 1)
    
    
    mat1 = TrussMaterial(E,A,rho)
    mat2 = TrussMaterial(alpha*E,A,beta*rho)
    #element1 = CrisfieldElement([node1,node2],timeOrder,mat2,1)
    #element2 = CrisfieldElement([node2,node3],timeOrder,mat1,1)
    element1 = NonlinearCrisfieldElement([node1,node2],timeOrder,mat2,1)
    element2 = NonlinearCrisfieldElement([node2,node3],timeOrder,mat1,1)
    #node3.setLoad(-E*A*H**3/(3.0*math.sqrt(3.0)*(element2.L**3)),1)
    
    mesh = TrussMesh(2)
    mesh.addNode(node1)
    mesh.addNode(node2)
    mesh.addNode(node3)
    mesh.addElement(element1)
    mesh.addElement(element2)
    mesh.generateID()
    return mesh

timeOrder = 2    
mesh = build_structure(2.0e7,1.0,1.0,0.7,1.0,0.1,100,100,timeOrder)
mesh.Nodes[1].setU([0.050633741721625,0.0])
mesh.Nodes[2].setU([0.0,0.574573486600207])
mesh.Nodes[1].setA([0.041777453220745e6,0.0])
mesh.Nodes[2].setA([0.0,-6.650981196224420e6])

# Output
class plotOutput(FO.FEMOutput):
    def __init__(self, Nstep, Neq, timeOrder):
        self.Nsetp = Nstep
        self.Neq = Neq
        self.U = np.zeros((Neq,Nstep))
        self.timeOrder = timeOrder
        if timeOrder > 0:
            self.V = np.zeros((Neq,Nstep))
        if timeOrder == 2:
            self.A = np.zeros((Neq,Nstep))
            
    def getU(self, istep=0):
        if self.U.ndim == 1:
            return self.U
        return self.U[:,istep]
        
    def getV(self, istep=0):
        if self.timeOrder >0:
            if self.V.ndim == 1:
                return self.V
            return self.V[:,istep]
            
    def getA(self, istep=0):
        if self.timeOrder ==2:
            if self.A.ndim == 1:
                return self.A
            return self.A[:,istep]
        
    def outputData(self,data):
        if self.timeOrder == 0:
            if data.getTimeOrder() > 0:
                self.U[:,data.getCurrentStep()] = data.getU()
            else:
                self.U = data.getU()
        if self.timeOrder > 0:
            self.U[:,data.getCurrentStep()-1] = data.getU()
            self.V[:,data.getCurrentStep()-1] = data.getV()
            if self.timeOrder == 2:
                self.A[:,data.getCurrentStep()] = data.getA()

class StructureNonlinearNewmark(NM.NonlinearAlphaAlgorithm):
    def calculateREffect(self):
        """
        effective load vector
        """
        self.Ri *= -1.0
        self.Ri += self.Re
        if self.timeOrder > 0:
            np.dot(self.D,self.V,self.__tempR__)
            self.Ri -= self.__tempR__
        if self.timeOrder == 2:
            np.dot(self.M,self.A,self.__tempR__)
            self.Ri -= self.__tempR__

# Algorithm
Nstep = 200
time = 2.0e-2
output = plotOutput(Nstep,mesh.getNeq(),1)
#static = FA.LinearStaticAlgorithm(mesh,output,sv.numpySolver())
#static.calculate()
#mesh.plot()
#mesh.updateValues(output.getU(),None,None)
#mesh.plot(init=False,col='r')
#alg = NM.LinearNewmarkAlgorithm(mesh,2,output,sv.numpySolver(),time,Nstep,1.0)
alg = StructureNonlinearNewmark(mesh,2,output,sv.numpySolver(),time,Nstep,1.0)
alg.calculate()
tarr = np.array(list(range(0,200)))*2.0e-2/200
pl.plot(tarr,output.U[0,:],'b',tarr,output.U[1,:],'r')
    