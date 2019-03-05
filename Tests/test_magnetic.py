# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:57:29 2017

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
import Algorithm.NewmarkAlgorithm as NM
import Math.Solver as sv
import Math.IntegrationData as idat
import Mesh.MeshGenerator as mg
import cProfile
import pstats
import Algorithm.NewtonRaphson as NR
        
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

class JAMaterial(mat.Material):
    def __init__(self, sigma, ng, idx):
        self.a = 0.0
        self.alpha = 0.0
        self.c = 0.0
        self.Ms = 0.0
        self.k = 0.0
        self.delta = 1
        self.sigma = sigma
        self.eps = 8.854187817e-12
        self.dM = np.zeros(2)
        self.mu0 = 4.0e-7*np.pi
        self.ng = ng
        self.Bprev = np.zeros((2,ng))
        self.Mprev = np.zeros((2,ng))
        self.dMprev = np.zeros((2,ng))
        self.Hprev = np.zeros((2,ng))
        self.idx = idx
        self.Mu = np.zeros(2)
        self.Ndof = 1
        self.hystereis = True
        
    def getID(self):
        return self.idx
        
    def updateParamters(self, a, alpha, c, Ms, k):
        self.a = a
        self.alpha = alpha
        self.c = c
        self.Ms = Ms
        self.k = k
        
    def calculateParameters(self, temperature):
        t0 = temperature/1.0213513430455913e3
        self.Ms = 1.6666270496980909e6*math.pow((1.0-t0),2.0588027319169142e-1)
        self.a = math.exp(1.1065973379588542e1)*\
        math.pow((1-t0),1.7544087504777564e-1)
        self.alpha=math.exp(-2.7711734827753376e0)*\
        math.pow((1-t0),-1.1702805122223958e-1)
        self.c = math.exp(-1.339064360358903e0)*\
        math.pow((1-t0),-3.4877155040447291e-2)
        self.k = math.exp(8.8017026926921e0)*\
        math.pow((1-t0),2.4926461785971135e-1)
        
    def calculateOne(self,data,b,ix):
        try:
            T = data.getU()[1]
        except:
            T = 298.0
        self.calculateParameters(T)
        try:
            ig = data.ig
        except:
            ig = 0
        if T>=1.02407149938785e3:
            self.Mu[ix,ig] = 0.0
            self.dM[ix,ig] = 0.0
            return
            
        nstep = 400
        try:
            Bprev = self.Bprev[ix,ig]
        except IndexError:
            print(ix,data.ig)
        Hprev = self.Hprev[ix,ig]
        Mprev = self.Mprev[ix,ig]
        if(b < Bprev):
            self.delta = -1
        else:
            self.delta = 1
            
        deltab = (b - Bprev)/nstep
        barr = Bprev
        h = Hprev
        mu1 = Mprev
        
        if math.fabs(deltab) > 0:
            for i in range(nstep):
                he = h + self.alpha*mu1
                man = self.Ms*Langevin(he/self.a)
                try:
                    dman = self.Ms/self.a*dLangevin(he/self.a)
                except:
                    print(Bprev,Hprev,barr,h,b,mu1,self.dMprev,he/self.a)
                    raise
                dman = dman/(1.0-self.alpha*dman)
                c1 = 1.0/(1.0+self.c)
                dmu1 = c1*(man-mu1)/(self.delta*self.k-\
                self.alpha*(man-mu1))+self.c*c1*dman
                if dmu1 <0:
                    dmu1 = -dmu1
                dmu1 = dmu1/(self.mu0*(1.0+dmu1))
                mu1 = mu1 + dmu1*deltab
                
                barr = barr + deltab
                h = barr/self.mu0 - mu1
                self.Mu[ix] = mu1
                self.dM[ix] = dmu1
        else:
            self.Mu[ix] = Mprev
            self.dM[ix] = self.dMprev[ix,ig]
            
        try:
            if data.store:
                self.Bprev[ix,ig] = b
                self.Hprev[ix,ig] = h
                self.Mprev[ix,ig] = self.Mu[ix]
                self.dMprev[ix,ig] = self.dM[ix]
        except AttributeError:
            self.Bprev[ix,ig] = b
            self.Hprev[ix,ig] = h
            self.Mprev[ix,ig] = self.Mu[ix]
            self.dMprev[ix,ig] = self.dM[ix]
            
        return self.Mu[ix]
            
    def calculate(self,data):
        B = data.getB()
        self.calculateOne(data,B[0],0)
        self.calculateOne(data,B[1],1)
        
    def calculateX(self, b):
        return self.calculateOne(None,b,0)
        
def Langevin(x):
    n = 8
    if math.fabs(x)>1.0:
        return 1.0e0/math.tanh(x) - 1.0e0/x
    else:
        g = 0.0
        for k in range(n,1,-1):
            bk = 2.0*k + 1.0
            g = x*x/(bk + g)
        return x/(3.0 + g)
        
def dLangevin(x):
    if math.fabs(x) < 1.0e-5:
        return 1.0e0/3.0e0
    else:
        a = math.sinh(x)
        return 1.0/(x*x)-1.0/(a*a)
    
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
        
    def calculateK(self, K, inod, jnod, t):
        r = self.x_[0]
        # magnetization
        if self.material.hysteresis:
            dNs = self.dN_
            Ns = self.N_
            dm1 = self.material.dM[0]
            dm2 = self.material.dM[1]
            ke = dm2*dNs[0][inod]*dNs[0][jnod];
            ke += dm2*Ns[inod]*dNs[0][jnod]/r;
            ke += dm2*dNs[0][inod]*Ns[jnod]/r;
            ke += dm2*Ns[inod]*Ns[jnod]/(r*r);
            ke += dm1*dNs[1][inod]*dNs[1][jnod];
            ke *= self.getFactor()
            K[0,0] -= ke
    
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
        re = self.N_[inod]*self.getBodyLoad(t)
        re *= self.getFactor()
        R[0] = re
    

def readInput(filename,nodeOrder,timeOrder,intData,Ndof = 1):
    mesh = FM.Mesh()
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
            nodes.append(findNode(mesh.getNodes(),a[j+1]-1))
        e = AxiSymMagnetic(nodes,[2,2],\
        QE.LagrangeBasis1D,nodeOrder,None,intData)
        mesh.addElement(e)
    file.readline()
    for i in range(nnod):
        a = list(float(x) for x in file.readline().split())
        for j in range(Ndof):
            mesh.getNodes()[i].setLoad(a[j+1],j)
            
    file.readline()
    for i in range(nnod):
        a = file.readline().split()
        for j in range(Ndof):
            mesh.getNodes()[i].setConstraint(int(a[2*j+1+2])==0,float(a[2*(j+1)+2]),j)
            
    air = LinearMagneticMaterial(1.0,1.0,0.0,2)
    cooper = LinearMagneticMaterial(1.0,1.0,5.0e6,3)
    steel = LinearMagneticMaterial(100.0,1.0,5.0e6,1)
    #steel = JAMaterial(5.0e6,9,1)
    file.readline()
    for i in range(nelm):
        a = list(int(x) for x in file.readline().split())
        if a[1] == 2:
            mesh.getElements()[i].updateMat(air)
        if a[1] == 3:
            mesh.getElements()[i].updateMat(cooper)
        if a[1] == 1:
            mesh.getElements()[i].updateMat(steel)
            #mesh.getElements()[i].setLinearity(False)
            #mesh.getElements()[i].updateMat(JAMaterial(5.0e6,intData.getNumberPoint(),1))
    file.close()
    return mesh
    
        
        
def findNode(nodes,id_number):
    for node in nodes:
        if node.get_id_number() == id_number:
            return node
    raise Exception()

Ndof = 1             
tOrder = 0
Ng = [3,3]
totalTime = 1.0e-3
numberTimeSteps = 100
rho_inf = 0.9
tol = 1.0e-8
load = 355.0/0.015/0.01
#load = 355.0

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)

nodeOrder = [[0,1,2,0,1,2,0,1,2],
             [0,0,0,1,1,1,2,2,2]]

#mesh = readInput('/home/haiau/Documents/testfortran_.dat',nodeOrder,tOrder,intDat)
#for e in mesh.getElements():
#    if e.material.getID() == 3:
#        def loadfunc(x,t):
#                return -load*math.sin(8.1e3*2*np.pi*t)
#        e.setBodyLoad(loadfunc)

def loadfunc(x,t):
    #return -load*math.sin(8.1e3*2*np.pi*t)
    return -load

def create_mesh():
    nodes = []
    nodes.append(FN.Node([0.0,-0.2],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.015,-0.2],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0225,-0.2],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.0325,-0.2],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([0.2,-0.2],Ndof,timeOrder = tOrder))
    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [0.1,0.05,0.014,0.015,0.0135,0.015,0.0135,0.015,0.014,0.05,0.1]
    #s = [0.1,0.064,0.015,0.0135,0.015,0.0135,0.015,0.064,0.1]
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
    
    polys = geo.getPolygons()
    for i in range(11):
        polys[i].setDivisionEdge13(4)
        
    for i in range(11,22):
        polys[i].setDivisionEdge13(2)
        
    for i in range(33,44):
        polys[i].setDivisionEdge13(5)
        
    for i in range(0,34,11):
        polys[i].setDivisionEdge24(4)
        
    for i in range(1,35,11):
        polys[i].setDivisionEdge24(2)
        
    for i in range(9,43,11):
        polys[i].setDivisionEdge24(2)
        
    for i in range(10,44,11):
        polys[i].setDivisionEdge24(4)
        
    mat3 = LinearMagneticMaterial(1.0,1.0,5.0e6,3)
    mat2 = LinearMagneticMaterial(1.0,1.0,0.0,2)
    #mat1 = JAMaterial(5.0e6,9,1)
    mat1 = LinearMagneticMaterial(1.0,1.0,5.0e6,1)
    for i in range(1,10):
        polys[i].setMaterial(mat1)
        
    polys[25].setMaterial(mat3)
    polys[25].setBodyLoad(load)
    polys[27].setMaterial(mat3)
    polys[27].setBodyLoad(load)
    polys[29].setMaterial(mat3)
    polys[29].setBodyLoad(load)
    
    for poly in polys:
        if poly.getMaterial() is None:
            poly.setMaterial(mat2)
        
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
        
    #fig = geo.plot(poly_number = True, fill_mat = True)
        
#    geo.plotMesh(col = 'b-',fill_mat = True)
#    for i,node in enumerate(nodesx):
#        pl.plot(node.getX()[0],node.getX()[1],'.b')
#        if math.fabs(node.getX()[0] - 0.015)<1.0e-14:
#            pl.text(node.getX()[0],node.getX()[1],str(i))
       
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14 or \
        math.fabs(n.getX()[1]-0.2)<1.0e-14 or \
        math.fabs(n.getX()[0]-0.2)<1.0e-14 or \
        math.fabs(n.getX()[1]+0.2)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            #n.setConstraint(False, 0.0, 1)
            #pl.plot(n.getX()[0],n.getX()[1],'.r')
    
    elements = []
    for i,e in enumerate(elems):
        #if mats[i] is JAMaterial:
        #    m = JAMaterial(5.0e6,9,1)
        #else:
        #    m = mats[i]
        m = mats[i]
        #elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        #QE.generateQuadNodeOrder([2,2],2),m,intDat))
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        nodeOrder,m,intDat))
        #if m.getID() == 1:
        #    elements[-1].setLinearity(False)
        #if bdls[i] is not None:
        if elements[-1].material.getID() == 3:
            elements[-1].setBodyLoad(loadfunc)
        
        
    mesh =  FM.Mesh()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
        
    return mesh

def create_simple_mesh():
    nodes = []
    nodes.append(FN.Node([0.0,0.0],Ndof,timeOrder = tOrder))
    nodes.append(FN.Node([1.0,0.0],Ndof,timeOrder = tOrder))
    
    edge = mg.Edge(nodes[0],nodes[1])
    poly = edge.extendToQuad(np.array([0.0,1.0]),1.0)
    
    
    geo = mg.Geometry()
    geo.addPolygon(poly)
    
    poly.setDivisionEdge13(4)
    
    mat2 = LinearMagneticMaterial(100.0,1.0,5.0e6,2)
    poly.setMaterial(mat2)
    
    
    geo.mesh()
    [nodesx, elems, mats, bdls] = geo.getMesh(None,mg.nodesQuad9,2)
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
        if math.fabs(n.getX()[1])<1.0e-14 or math.fabs(n.getX()[1]-1.0)<1.0e-14:
            n.setConstraint(False, 10*0.5*n.getX()[0],0)
        if math.fabs(n.getX()[0]-1.0)<1.0e-14:
            n.setConstraint(False, 10*0.5*n.getX()[0],0)
    elements = []
    for i,e in enumerate(elems):
        m = mats[i]
        elements.append(AxiSymMagnetic(e,[2,2],QE.LagrangeBasis1D,\
        QE.generateQuadNodeOrder([2,2],2),m,intDat))
        
    mesh =  FM.Mesh()
    mesh.addNodes(nodesx)
    mesh.addElements(elements)
    
    def loadfunc(t):
        #return load*math.cos(8.1e3*2*np.pi*t)
        #return load*math.cos(8.1e3*2*np.pi*t)
        return load
    
#    mesh.Nodes[4].setLoad(loadfunc,0)
    
    return mesh
        

    
#mesh = create_simple_mesh()
        
#matx = JAMaterial(0.0,1,1)
#btest1 = [0.02*i for i in range(100)]+[0.02*i for i in range(100,-1,-1)]
#btest = np.array(btest1)
#mtest = np.array([matx.calculateX(b) for b in btest])
#pl.plot((btest/(np.pi*4.0e-7)-mtest),mtest)

#mesh = create_mesh()
mesh = create_simple_mesh()

#cnt = 0
#for e in mesh.Elements:
#    if e.bodyLoad is not None:
#        cnt += 1

#mesh.plot(fill_mat = True)

mesh.generateID()      

output = FO.StandardFileOutput('/home/haiau/Documents/result.dat')
#alg = NM.NonlinearNewmarkAlgorithm(mesh,tOrder,output,sv.numpySolver(),\
#totalTime, numberTimeSteps,rho_inf,tol=1.0e-8)

alg = NR.LoadControlledNewtonRaphson(mesh,output,sv.numpySolver(),1)

alg.calculate()

#output.updateToMesh(mesh,10)
#X,Y,Z = mesh.meshgridValue([0.0,0.2,-0.2,0.2],0.01,1.0e-8)
#
#cProfile.run('alg.calculate()','calculate.profile')
#stats = pstats.Stats('calculate.profile')
#stats.strip_dirs().sort_stats('time').print_stats()
#
#
#_,inod = mesh.findNodeNear(np.array([0.015,0.0]))
#testout,tout = output.readOutput('/home/haiau/Documents/result.dat',list(range(50)),inod,'v')
#testout = [t[0][0] for t in testout]
