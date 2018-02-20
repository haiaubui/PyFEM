# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:20:19 2017

@author: haiau
"""

import math
import numpy as np
import pylab as pl
import FEMBoundary as FB
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

class AxiSymMagneticBoundary(AE.AxisymmetricStaticBoundary,FB.StraightBoundary1D):
    def calculateKLinear(self, K, i, j, t):
        K[0,1] = self.N_[i]*self.N_[j]
        K[0,1] *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
        K[1,0] = -self.N_[i]*self.N_[j]*0.5
        K[1,0] *= self.getFactor()
        
#    def calculateR(self, R, i, t):
#        r0 = self.N_[i]*self.u_[1]
#        r0 *= self.getFactor()*2.0*np.pi*self.x_[0]/self.material.mu0
#        r1 = -self.N_[i]*self.u_[0]*0.5
#        r1 *= self.getFactor()
#        R[0] += r0
#        R[1] += r1
        
        
        
Ndof = 2             
tOrder = 1
Ng = [3,3]
totalTime = 1.0e-3
numberTimeSteps = 100
rho_inf = 0.9
tol = 1.0e-8
load = 355.0/0.015/0.01

intDat = idat.GaussianQuadrature(Ng, 2, idat.Gaussian1D)

intDatB = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)
#intSingDat = idat.GaussianQuadrature(3, 1, idat.Gaussian1D)

# converged solution only with special Gaussian quadrature
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
    ndiv = 1
        
    polys = geo.getPolygons()
#    for p in polys:
#        p.setDivisionEdge13(ndiv)
#        p.setDivisionEdge24(ndiv)
    for i in range(9):
        polys[i].setDivisionEdge13(4*ndiv)
    
    polys[0].setDivisionEdge24(2*ndiv)
    polys[8].setDivisionEdge24(2*ndiv)
    
    mat2 = LinearMagneticMaterial(1.0,0.0,5.0e6,2)
    #mat1 = JAMaterial(5.0e6,9,1)
    mat1 = LinearMagneticMaterial(100.0,0.0,5.0e6,1)
    
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
        #if m.getID() == 1:
        #    elements[-1].setLinearity(False)
        if bdls[i] is not None:
            def loadfunc(x,t):
                return load*math.sin(8.1e3*2*np.pi*t)
                #return -load
        else:
            loadfunc = None
        elements[-1].setBodyLoad(loadfunc)
        
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
        elementBs[-1].setMaterial(mat2)
        
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

output = FO.StandardFileOutput('/home/haiau/Documents/result_linear_ndiv_4_3200.dat')
alg = NM.NonlinearNewmarkAlgorithm(mesh,tOrder,output,sv.numpySolver(),\
totalTime, numberTimeSteps,rho_inf,tol=1.0e-8)

#output.chooseSteps(range(0,100,2))

alg.calculate()

_,inod = mesh.findNodeNear(np.array([0.015,0.0]))
testout,tout = output.readOutput('/home/haiau/Documents/result_linear_ndiv_4_3200.dat',list(range(50)),inod,'u')
testout = [t[0][0] for t in testout]

#output.updateToMesh(mesh,10)
##X,Y,Z = mesh.meshgridValue([0.0,0.2,-0.2,0.2],0.01,1.0e-8)
#X,Y,Z = mesh.meshgridValue([0.0,3.0,-2.0,3.0],0.02,1.0e-8)
#udat = [n.getU().tolist() for n in mesh.Nodes]
####Test Element######
#nodes = []
#nodes.append(FN.Node([-1.0,-1.0],2))
#nodes.append(FN.Node([0.0,-1.0],2))
#nodes.append(FN.Node([1.0,-1.0],2))
#nodes.append(FN.Node([-0.75,0.0],2))
#nodes.append(FN.Node([0.0,0.0],2))
#nodes.append(FN.Node([0.75,0.0],2))
#nodes.append(FN.Node([-0.5,1.0],2))
#nodes.append(FN.Node([0.0,1.0],2))
#nodes.append(FN.Node([0.5,1.0],2))
#
#teste = AxiSymMagnetic(nodes,[2,2],QE.LagrangeBasis1D,\
#        QE.generateQuadNodeOrder([2,2],2),None,intDat)
#
#N_ = np.zeros(teste.Nnod,teste.dtype)
#dN_ = np.zeros((teste.Ndim,teste.Nnod),teste.dtype)
#
#testxi = teste.getXi(np.array([0.625,0.5]))
#print(testxi)
#teste.basisND(testxi,N_,dN_)
#testx = np.zeros(2,teste.dtype)
#teste.getX(testx,N_)
#print(testx)
#
#node1 = FN.Node([0.015, -0.1],2)
#node2 = FN.Node([0.015, -0.08750000000000001],2)
#node3 = FN.Node([0.015, -0.07500000000000001],2)
#
#testb = AxiSymMagneticBoundary([node1,node2,node3],2,\
#        QE.LagrangeBasis1D,QE.generateQuadNodeOrder(2,1),\
#        intDatB,intSingDat,[0.0,1.0],0)
#
#Nb_ = np.zeros(testb.Nnod,testb.dtype)
#dNb_ = np.zeros((testb.Ndim,testb.Nnod),testb.dtype)
#
#testxib = testb.getXi(np.array([0.625,0.5]))
#print(testxib)
#testb.basisND(testxib,Nb_,dNb_)
#testxb = np.zeros(2,testb.dtype)
#testb.getX(testxb,Nb_)
#print(testxb)
