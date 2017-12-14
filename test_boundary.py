# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:20:19 2017

@author: haiau
"""

import AxisymmetricElement as AE


import math
import numpy as np
import pylab as pl
import AxisymmetricElement as AE
import QuadElement as QE
import FEMNode as FN
import FEMMesh as FM
import FEMOutput as FO
import Material as mat
import NewmarkAlgorithm as NM
import Solver as sv
import IntegrationData as idat
import SingularIntegration as SI
import MeshGenerator as mg
import cProfile
import pstats
        
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
        
class AxiSymMagnetic(AE.AxisymmetricQuadElement):
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
        # magnetization
        if self.material.hysteresis:
            dm1 = self.material.dM[0]
            dm2 = self.material.dM[1]
            K[0,0] -= dm2*self.dN_[0,inod]*self.dN_[0,jnod]
            K[0,0] -= dm2*self.N_[inod]*self.dN_[0,jnod]/r
            K[0,0] -= dm2*self.dN_[0,inod]*self.N_[jnod]/r
            K[0,0] -= dm2*self.N_[inod]*self.N_[jnod]/(r*r)
            K[0,0] -= dm1*self.dN_[1,inod]*self.dN_[1,jnod]
        K[0,0] *= self.getFactor()
    
    def calculateDLinear(self, D, inod, jnod, t):
        """
        Calculate Damping matrix D
        """
        D[0,0] = self.N_[inod]*self.N_[jnod]
        D *= self.material.sigma*self.getFactor()
    
    def calculateMLinear(self, M, inod, jnod, t):
        """
        Calculate Mass matrix M
        """
        M[0,0] = self.N_[inod]*self.N_[jnod]
        M *= self.material.eps*self.getFactor()
    
    def calculateR(self, R, inod, t):
        """
        Calculate load matrix R
        """
        r = self.x_[0]
        re = 0.0
        if self.material.hysteresis:
            re += self.material.Mu[0]*self.dN_[1,inod]
            re -= self.material.Mu[1]*(self.N_[inod]/r+self.dN_[0,inod])
        R[0] += re
        
    def calculateRe(self, R, inod, t):
        re = -self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R[0] = re
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None):
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

class AxiSymMagneticBoundary(AE.AxisymmetricStaticBoundary):
    def calculateKLinear(self, K, i, j, t):
        K[0,1] = self.N_[i]*self.N_[j]
        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
        K[1,0] = -self.N_[i]*self.N_[j]*0.5
        K[1,0] *= self.getFactor()
        
    def calculateR(self, R, i, t):
        r0 = self.N_[i]*self.u_[1]
        r0 *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
        r1 = -self.N_[i]*self.u_[0]*0.5
        r1 *= self.getFactor()
        R[0] += r0
        R[1] += r1
        
        
        
Ndof = 2             
tOrder = 2
Ng = [3,3]
totalTime = 2.0
numberTimeSteps = 2000
rho_inf = 0.9
tol = 1.0e-8
load = 355.0/0.015/0.01

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)

intDatB = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)
#intSingDat = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)
intSingDat = SI.SingularGaussian1D(12, intDatB.xg,\
SI.Gaussian_1D_Pn_Log, SI.Gaussian_1D_Pn_Log_Rat)
#intSingDat = None

def readInput(filename,nodeOrder,timeOrder,intData,Ndof = 1):
    mesh = FM.MeshWithBoundaryElement()
    file = open(filename,'r')
    int(file.readline().split()[1])
    nnode = int(file.readline().split()[1])
    nnod = int(file.readline().split()[1])
    nelm = int(file.readline().split()[1])
    int(file.readline().split()[1])
    int(file.readline().split()[1])
    file.readline()
    for i in range(nnod):
        a = list(float(x) for x in file.readline().split())
        x_ = np.array(a[1:3])
        mesh.addNode(FN.Node(x_,Ndof,timeOrder,i))
    file.readline()
    for i in range(nelm):
        a = list(int(x) for x in file.readline().split())
        nodes = []
        for j in range(nnode):
            try:
                nn = findNode(mesh.getNodes(),a[j+1]-1)
                nodes.append(nn)
            except:
                continue
        if len(nodes) > 3:
            e = AxiSymMagnetic(nodes,[2,2],\
            QE.LagrangeBasis1D,nodeOrder,None,intData)
            mesh.addElement(e)
        elif len(nodes) == 3:
            e = AxiSymMagneticBoundary(nodes,2,QE.LagrangeBasis1D,\
            QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,None,i)
            for el in mesh.getElements():
                if el.hasNodes(nodes):
                    edge = mg.Edge(nodes[0],nodes[2])
                    e.setNormalVector(\
                    edge.getNormalVector(el.getNodes()[4].getX()))
                    break
            mesh.addBoundaryElement(e)
        
    file.readline()
    for i in range(nnod):
        a = list(float(x) for x in file.readline().split())
        #for j in range(Ndof):
        #    mesh.getNodes()[i].setLoad(a[j+1],j)
            
    file.readline()
    for i in range(nnod):
        a = file.readline().split()
        for j in range(Ndof):
            mesh.getNodes()[i].setConstraint(\
            int(a[2*j+1])==0,float(a[2*(j+1)]),j)
            
    air = LinearMagneticMaterial(1.0,1.0,3.4e7,2)
    cooper = LinearMagneticMaterial(1.0,1.0,5.0e6,3)
    steel = LinearMagneticMaterial(1.0,1.0,0.0,1)
    file.readline()
    for i in range(nelm):
        a = list(int(x) for x in file.readline().split())
        if a[1] == 2:
            mesh.getElements()[i].setMaterial(air)
        if a[1] == 3:
            mesh.getElements()[i].setMaterial(cooper)
        if a[1] == 1:
            mesh.getElements()[i].setMaterial(steel)
            if mesh.getElements()[i].getNodes()[0].getX()[0] < 0.06:
                def loadfunc(x,t):
                    return 20.0*960.0/(0.028*0.052)*math.sin(50.0*2.0*np.pi*t)
                mesh.getElements()[i].setBodyLoad(loadfunc)
            else:
                def loadfuncx(x,t):
                    return -20.0*576.0/(0.015*0.052)*math.sin(50.0*2.0*np.pi*t)
                mesh.getElements()[i].setBodyLoad(loadfuncx)
            #mesh.getElements()[i].updateMat(JAMaterial(5.0e6,intData.getNumberPoint(),1))
    file.close()
    mesh.generateID()
    return mesh
    
        
        
def findNode(nodes,id_number):
    for node in nodes:
        if node.get_id_number() == id_number:
            return node
    raise Exception()
    
def create_mesh():
    nodes = []
    nodes.append(FN.Node([0.0,-0.1],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.015,-0.1],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0225,-0.036],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0325,-0.036],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0225,-0.0075],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0325,-0.0075],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0225,0.021],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0325,0.021],Ndof,timeOrder = tOrder))
    
    edges = []
    edges.append(mg.Edge(nodes[0],nodes[1]))
    edges.append(mg.Edge(nodes[2],nodes[3]))
    edges.append(mg.Edge(nodes[4],nodes[5]))
    edges.append(mg.Edge(nodes[6],nodes[7]))

    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [0.05,0.014,0.015,0.0135,0.015,0.0135,0.015,0.014,0.05]
    geo.addPolygons(edges[0].extendToQuad(d,s))
    s = [0.2,0.015,0.015,0.015]
    for i in range(1,len(edges)):
        geo.addPolygons(edges[i].extendToQuad(d,s[i]))
        
    #fig = geo.plot(poly_number=True)
        
    polys = geo.getPolygons()
    for i in range(9):
        polys[i].setDivisionEdge13(4)
    
    polys[0].setDivisionEdge24(2)
    polys[8].setDivisionEdge24(2)
    
    mat2 = LinearMagneticMaterial(1.0,1.0,5.0e6,2)
    #mat1 = JAMaterial(5.0e6,9,1)
    mat1 = LinearMagneticMaterial(100.0,1.0,5.0e6,1)
    
    for i in range(9):
        polys[i].setMaterial(mat1)
    polys[9].setMaterial(mat2)
    polys[9].setBodyLoad(1.0)
    polys[10].setMaterial(mat2)
    polys[10].setBodyLoad(1.0)
    polys[11].setMaterial(mat2)
    polys[11].setBodyLoad(1.0)
    
    geo.mesh()
    
    #fig = geo.plotMesh(fill_mat = True)
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
    
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
        if bdls[i] is not None:
            def loadfunc(x,t):
                return load*math.sin(50.0*2*np.pi*t)
                #return load
        else:
            loadfunc = None
        elements[i].setBodyLoad(loadfunc)
        
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    elementBs = []
    for i,e in enumerate(elems1):
        elementBs.append(AxiSymMagneticBoundary(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(mat1)
        
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 1)
        
    #mesh.addElements(elementBs)
    mesh.addBoundaryElements(elementBs)
    mesh.generateID()

    
    #mesh.Nodes[4].setLoad(loadfunc,0)
    
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
    
    geo = mg.Geometry()
    geo.addPolygon(poly1)
    geo.addPolygon(poly2)
    
    mat1 = LinearMagneticMaterial(1.0,0.0,5.0e6,1)
    mat2 = LinearMagneticMaterial(1.0,0.0,0.0,2)
    poly1.setMaterial(mat1)
    poly2.setMaterial(mat2)
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,Ndof)
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
    elements = []
    
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
        if bdls[i] is not None:
            def loadfunc(x,t):
                return load*math.sin(50.0*2*np.pi*t)
                #return load
        else:
            loadfunc = None
        elements[i].setBodyLoad(loadfunc)
        
    mesh =  FM.MeshWithBoundaryElement()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    geo.meshBoundary()
    [nodes1, elems1, normBndVec] = geo.getBoundaryMesh(None,\
    mg.nodesBoundaryQuad9,Ndof)
    
    elementBs = []
    for i,e in enumerate(elems1):
        elementBs.append(AxiSymMagneticBoundary(e,2,QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder(2,1),intDatB,intSingDat,normBndVec[i],i))
        elementBs[-1].setMaterial(mat1)
        
    for n in mesh.getNodes():
        ine = False
        for e in elementBs:
            if n in e:
                ine = True
        if not ine:
            n.setConstraint(False, 0.0, 1)
        
    #mesh.addElements(elementBs)
    mesh.addBoundaryElements(elementBs)
    mesh.generateID()

    
    #mesh.Nodes[4].setLoad(loadfunc,0)
    
    return mesh
    
def get_IT(mesh):
    IT = []
    nodes = mesh.Nodes
    for e in mesh.Elements:
        IT.append([])
        for n in e.Nodes:
            for i in range(mesh.Nnod):
                if n == nodes[i]:
                    IT[-1].append(i+1)
                    
    return IT
    
#mesh = create_simple_mesh()
    
mesh = create_mesh()

#nodeOrder = [[2,1,0,2,1,0,2,1,0],
#             [2,2,2,1,1,1,0,0,0]]
#mesh = readInput('/home/haiau/Dropbox/Static_magnetic/testfortran_body.dat',\
#nodeOrder,tOrder,intDat,2)

#output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')
#alg = NM.NonlinearAlphaAlgorithm(mesh,tOrder,output,sv.numpySolver(),\
#totalTime, numberTimeSteps,rho_inf,tol=1.0e-8)
#
#alg.calculate()