# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 13:59:48 2018

@author: haiau
"""
import Mesh.MeshGenerator as MG
import Mesh.FEMNode as FN
import Element.FEMElement as FE
import Element.FEMBoundary as FB
import Element.TriangularElement as TE
import numpy as np
import pylab as pl
import math

class PolarNode(FN.Node):
    """
    Node in polar coordinates
    """
    
    def __eq__(self, node):
        """
        Compare this node to other node
        Two nodes are equal if they have same coordinates and number of dims
        If two nodes have different number of dimensions,
        raise NodesNotSameDimension
        Notice: two nodes a,b will be considered to be equal
        if |a-b|<1.0e-13
        """
        if self.Ndim != node.getNdim():
            raise MG.NodesNotSameDimension
            
        phi1 = reduce_2pi(self.X_[1])
        phi2 = reduce_2pi(node.getX()[1])
        r1 = self.X_[0]
        r2 = node.getX()[0]
        
        return math.sqrt((r1-r2)**2+(phi1-phi2)**2) < 1.0e-13
        
    def getX(self, loop = 0):
        """
        loop: number of times 2*pi is added
        Return the coordinates of this node, numpy array
        """
        if loop > 0:
            return np.array([self.X_[0],self.X_[1]+2.0*np.pi*loop])
        return self.X_
        

class PolarElement(FE.StandardElement):
    """
    Rectangular Element in Polar Coordinates
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, material, intData,\
    dtype = 'float64', commonData = None):
        """
        Initialize an Standard Element (in both 2-d and 3-d case)
        In 1-d case, this is a normal 1 dimensional element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunction: basis function, see super class info
            nodeOrder: the order of nodes in elements,
            for 1 dimensional element, nodeOrder is an increasing array,
            for >1 dimensional elelement, nodeOrder is a n-dim-array
            material: material of element
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
        """
        self.loop = [0]*len(Nodes)
        for i in range(pd[0]+1):
            for j in range(1,pd[1]+1):
                n1,c1 = FE.find_node_from_order(i,j,Nodes,nodeOrder)
                n0,c0 = FE.find_node_from_order(i,j-1,Nodes,nodeOrder)
                if n1.getX(self.loop[c1])[1] < n0.getX(self.loop[c0])[1]:
                    self.loop[c1] = 1
        FE.StandardElement.__init__(self,Nodes,pd,basisFunction,nodeOrder,\
        material,intData,dtype,commonData,ndime = 2)
     
    def getFactor(self):
        """
        Return: factor for integration, 2*pi*radius*det(J)
        """
        return self.x_[0]*self.factor[self.ig]
        
    def calculateBoundingBox(self):
        """
        Calculate bounding box of element
        """
        x = [n.getX(self.loop[i])[0] for i,n in enumerate(self.Nodes)]
        y = [n.getX(self.loop[i])[1] for i,n in enumerate(self.Nodes)]
        self.__maxx__ = max(x)
        self.__minx__ = min(x)
        self.__maxy__ = max(y)
        self.__miny__ = min(y)
        
    def insideBoundingBox(self, x):
        maxy = reduce_2pi(self.__maxy__)
        miny = reduce_2pi(self.__miny__)
        if miny > maxy:
            return x[0]<self.__maxx__+1.0e-13 and x[0]>self.__minx__-1.0e-13 \
            and reduce_2pi(x[1])+2.0*np.pi>miny-1.0e-13 \
            and reduce_2pi(x[1])<maxy+1.0e-13
        return x[0]<self.__maxx__+1.0e-13 and x[0]>self.__minx__-1.0e-13 \
        and reduce_2pi(x[1])<maxy+1.0e-13 and reduce_2pi(x[1])>miny-1.0e-13
    
    def Jacobian(self, dN_, edg=False):
        """
        Calculate det(J) and modify dN_
        return det(J)
        """
        __tempJ__ = np.zeros((self.Ndim,self.Ndim),self.dtype)
        
        Jmat = np.zeros((self.Ndim,self.Ndim),self.dtype)
        
        for i in range(self.Nnod):
#            x = np.array(self.Nodes[i].getX(self.loop[i]))
#            x[1] *= x[0]
            x = self.Nodes[i].getX(self.loop[i])
            np.outer(dN_[:,i],x,__tempJ__)
            Jmat += __tempJ__
        
        if edg:
            self.detEdg1 = math.sqrt(Jmat[0,0]**2+Jmat[1,0]**2)
            self.detEdg2 = math.sqrt(Jmat[0,1]**2+Jmat[1,1]**2)
        
        det = FE.__determinant__(Jmat)
        if np.allclose(det,0.0,rtol = 1.0e-14):
            for i in range(self.Nnod):
                a,b = 0.0,0.0
                if not np.allclose(Jmat[0,0],0.0,rtol=1.0e-14):
                    a = dN_[0,i]/Jmat[0,0]
                if not np.allclose(Jmat[0,1],0.0,rtol=1.0e-14):
                    b = dN_[0,i]/Jmat[0,1]
                dN_[0,i] = a
                dN_[1,i] = b
            return np.sqrt(Jmat[0,0]**2 + Jmat[0,1]**2)
        for i in range(self.Nnod):
            a = Jmat[0,0]*dN_[0,i]+Jmat[1,0]*dN_[1,i]
            b = Jmat[0,1]*dN_[0,i]+Jmat[1,1]*dN_[1,i]
            dN_[0,i] = a
            dN_[1,i] = b
        
        return det
        
    def getXi(self, x, N_ = None, dN_ = None, xi = None, max_iter = 100,\
    rtol = 1.0e-8):
        """
        Return natural coordinate xi corresponding to physical coordinate x
        N_: array to store shape functions
        dN_: array to store derivatives of shape functions
        max_iter: maximum number of iterations for Newton method
        Raise OutsideEdlement if x is not inside element.
        """
        if not self.insideBoundingBox(x):
            raise FE.OutsideElement
        
        if reduce_2pi(self.__miny__) > reduce_2pi(self.__maxy__):
            xtemp = np.array(x)
            x[1] += 2.0*np.pi
            
        if N_ is None:
            N_ = np.zeros(self.Nnod,self.dtype)
        if dN_ is None:
            dN_ = np.zeros((self.Ndim,self.Nnod),self.dtype)
            
        if xi is None:
            xi = np.empty(self.Ndime,self.dtype)
        xi.fill(0.0)
        __tempJ__ = np.empty((self.Ndim,self.Ndim),self.dtype)
        Jmat = np.empty((self.Ndim,self.Ndim),self.dtype)
        x0 = np.zeros(self.Ndim,self.dtype)
        #deltax = np.zeros(self.Ndime,self.dtype)
        
        for _ in range(max_iter):
            self.basisND(xi, N_, dN_)
            x0.fill(0.0)
            self.getX(x0, N_)
            Jmat.fill(0.0)
            for i in range(self.Nnod):
                xx = np.array(self.Nodes[i].getX(self.loop[i]))
                #xx[1] *= xx[0]
                np.outer(dN_[:,i],xx,__tempJ__)
#                np.outer(self.Nodes[i].getX(),dN_[:,i],__tempJ__)
#                #np.outer(dN_[:,i],self.Nodes[i].getX(),__tempJ__)
                Jmat += __tempJ__
            if np.allclose(FE.__determinant__(Jmat),0.0,rtol=1.0e-14):
                #x0 -= x
                if Jmat[0,0] != 0.0:
                    Jmat[0,0] = 1.0/Jmat[0,0]
                if Jmat[1,0] != 0.0:
                    Jmat[1,0] = 1.0/Jmat[1,0]
                deltax = np.dot((x0-x),Jmat[:,0])
            else:
                #x0 -= x
                #x0 *= -1.0
                deltax = np.dot(Jmat,(x0-x))
            xi -= deltax
            if np.linalg.norm(x0-x) < rtol:
                if np.any(np.fabs(xi) > 1.0+1.0e-13):
                    #print('outside',xi)
                    if reduce_2pi(self.__miny__) > reduce_2pi(self.__maxy__):
                        x = xtemp
                    raise FE.OutsideElement
                else:
                    if reduce_2pi(self.__miny__) > reduce_2pi(self.__maxy__):
                        x = xtemp
                    return xi
        if reduce_2pi(self.__miny__) > reduce_2pi(self.__maxy__):
            x = xtemp
        raise FE.OutsideElement
        
    def getX(self, x, N_ = None):
        """
        Get coordinates at parametric coordinates x (x from calculateBasis)
        """
        if N_ is None:
            N_ = self.N_
        
        try:
            for i in range(self.Nnod):
                x += self.Nodes[i].getX(self.loop[i])*N_[i]
        except TypeError:
            raise FE.ElementBasisFunctionNotCalculated
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None):
        """
        Default plot method
        This method simply plot the nodes continuously
        Derived classes should override this method to get desired shape
        """
        if fig is None:
            fig = pl.figure()
        N_,_ = self.basisND(np.array([0.0,0.0]))
        xt = np.zeros(self.Ndim,self.dtype)
        self.getX(xt,N_)
        if number is not None:
            pl.text(xt[1],xt[0],str(number))
        x = [n.getX()[0] for n in self.Nodes]
        if self.Ndim > 1:
            y = [n.getX()[1] for n in self.Nodes]
        #if self.Ndim == 3:
        #    z = [n.getX()[2] for n in self.Nodes]
        if self.Ndim == 1:
            return pl.polar(np.array(x),col),[n.getX() for n in self.Nodes]
        if self.Ndim == 2:
            return pl.polar(np.array(x),np.array(y),col),\
            [n.getX() for n in self.Nodes]
        if self.Ndim == 3:
            """
            Will be implemented later
            """
            return None

class PolarT6Element(TE.T6Element, PolarElement):
    """
    6 Nodes Triangular element in Polar coordinates
    """
    def __init__(self, Nodes, material, dtype = 'float64',\
    commonData = None, nG = 4):
        """
        Initialize an Standard Element (in both 2-d and 3-d case)
        In 1-d case, this is a normal 1 dimensional element
        Input:
            Nodes: nodes of elements
            material: material of element
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
            nG: number of Gaussian quadrature points
        """
        self.loop = [0]*len(Nodes)
        # find edge with nodes that have equal radius
        egs = [[0,4,1],[1,5,2],[2,3,0]]
        for es in egs:
            xn1 = Nodes[es[0]].getX()[0]
            xn2 = Nodes[es[1]].getX()[0]
            xn3 = Nodes[es[2]].getX()[0]
            if math.fabs(xn1-xn2) < 1.0e-13 and math.fabs(xn2-xn3) < 1.0e-13:
                xp1 = Nodes[es[0]].getX()[1]
                xp2 = Nodes[es[1]].getX()[1]
                xp3 = Nodes[es[2]].getX()[1]
                if xp1 > xp2:
                    self.loop[es[1]] = 1
                if xp2 > xp3:
                    self.loop[es[2]] = 1
                    
        TE.T6Element.__init__(self,Nodes,material,dtype,commonData,nG)
        
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None):
        if fig is None:
            fig = pl.figure()
        
        X1 = self.Nodes[0].getX()
        X2 = self.Nodes[1].getX()
        xa1 = np.linspace(min(X1[1],X2[1]),max(X1[1],X2[1]),20)
        xa2 = np.linspace(min(X1[0],X2[0]),max(X1[0],X2[0]),20)
        pl.polar(xa1,xa2,col)
        
        X1 = self.Nodes[1].getX()
        X2 = self.Nodes[2].getX()
        xa1 = np.linspace(min(X1[1],X2[1]),max(X1[1],X2[1]),20)
        xa2 = np.linspace(min(X1[0],X2[0]),max(X1[0],X2[0]),20)
        pl.polar(xa1,xa2,col)
        
        X1 = self.Nodes[2].getX()
        X2 = self.Nodes[0].getX()
        xa1 = np.linspace(min(X1[1],X2[1]),max(X1[1],X2[1]),20)
        xa2 = np.linspace(min(X1[0],X2[0]),max(X1[0],X2[0]),20)
        pl.polar(xa1,xa2,col)
        
        nodes = self.Nodes
        for n in nodes:
            pl.polar(n.getX()[1],n.getX()[0],'.b')
        
        if number is not None:
            c = 0.5*(nodes[4].getX()+nodes[5].getX())
            pl.text(c[0]*math.cos(c[1]),c[0]*math.sin(c[1]),str(number))
        
        return fig, [nodes[0],nodes[1],nodes[2]]

class PolarElementGapBoundary(FE.Element):
    def __init__(self, Nodes, pd, basisFunction, matConstant,\
    dtype = 'float64', nHarmonic = 10, rRatio = 0.5, normv = [1.0,0.0],\
    mechdof = -1):
        """
        Initialize an Standard Element (in both 2-d and 3-d case)
        In 1-d case, this is a normal 1 dimensional element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunction: basis function, see super class info
            nodeOrder: the order of nodes in elements,
            for 1 dimensional element, nodeOrder is an increasing array,
            for >1 dimensional elelement, nodeOrder is a n-dim-array
            material: material of element
            intData: integration data
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
            gapEdge: edge that is treated with solution from air gap
            nHarmonic: number of Harmonic for air gap solution
        """
        FE.Element.__init__(self,Nodes,pd,basisFunction,None,None,dtype,None)
        self.normv = np.array(normv)
        self.nHarmonic = nHarmonic
        self.Ndime = 1
        self.harmonicIntSin = np.zeros((self.pd+1,nHarmonic),dtype)
        self.harmonicIntCos = np.zeros((self.pd+1,nHarmonic),dtype)
        self.coeffA = np.zeros(nHarmonic,dtype)
        self.coeffB = np.zeros(nHarmonic,dtype)
        self.coeffC = np.zeros(nHarmonic,dtype)
        self.coeffD = np.zeros(nHarmonic,dtype)
        self.Nb_,self.dNb_ = self.basisFunc(-1.0,self.pd)
        self.Ne_,self.dNe_ = self.basisFunc(1.0,self.pd)
        self.mechdof = mechdof
        self.rRatio = rRatio
        self.matConstant = matConstant
        self.linear = True
        
        self.loop = [0]*self.Nnod
        for i in range(1,self.Nnod):
            phi1 = self.Nodes[i].getX(self.loop[i])[1]
            phi0 = self.Nodes[i-1].getX(self.loop[i-1])[1]
            if phi1 < phi0:
                self.loop[i] = 1
        
    def plot(self, fig = None, col = '-r', fill_mat = False, number = None):
        if fig is None:
            fig = pl.figure()
        
        X1 = self.Nodes[0].getX(self.loop[0])
        X2 = self.Nodes[-1].getX(self.loop[-1])
        xa1 = np.linspace(min(X1[1],X2[1]),max(X1[1],X2[1]),20)
        xa2 = np.linspace(min(X1[0],X2[0]),max(X1[0],X2[0]),20)
        pl.polar(xa1,xa2,col)
        
        nodes = self.Nodes
        for n in nodes:
            pl.polar(n.getX()[1],n.getX()[0],'.b')
        
        if number is not None:
            c = 0.5*(nodes[0].getX(self.loop[0])+nodes[-1].getX(self.loop[-1]))
            pl.text(c[1],c[0],str(number))
        
        return fig, [nodes[0],nodes[-1]]
        
    def getXi(self, x, N_ = None, dN_ = None, xi = None, max_iter = 100,\
    rtol = 1.0e-8):
        raise FE.OutsideElement
        
    def prepareElement(self):
        pass
    
    def calculateBodyLoad(self,data):
        pass
        
    def calculateIntegrals(self, mechdof = -1, deltaT = 0.0):
        self.harmonicIntSin.fill(0.0)
        self.harmonicIntCos.fill(0.0)
        thetaBeg = self.Nodes[0].getX()[1]
        thetaEnd = self.Nodes[-1].getX()[1]
        if self.current:
            if mechdof > -1:
                thetaBeg += self.Nodes[0].getU()[mechdof]
                thetaEnd += self.Nodes[-1].getU()[mechdof]
            if self.movingVel is not None:
                thetaBeg += self.movingVel[1]*deltaT
                thetaEnd += self.movingVel[1]*deltaT
        
        thetaBeg = reduce_2pi(thetaBeg)
        thetaEnd = reduce_2pi(thetaEnd)
        if thetaBeg > thetaEnd:
            thetaEnd += 2.0*np.pi
        deltaTheta =  thetaEnd - thetaBeg 
        if self.pd == 2:
            b0b = self.dNb_[0]*2.0/deltaTheta
            a0b = self.Nb_[0]
            c0b = 4.0/(deltaTheta**2)
            
            b0e = self.dNe_[0]*2.0/deltaTheta
            a0e = self.Ne_[0]
            c0e = 4.0/(deltaTheta**2)
            
            b1b = self.dNb_[1]*2.0/deltaTheta
            a1b = self.Nb_[1]
            c1b = -8.0/(deltaTheta**2)
            
            b1e = self.dNe_[1]*2.0/deltaTheta
            a1e = self.Ne_[1]
            c1e = -8.0/(deltaTheta**2)
            
            b2b = self.dNb_[2]*2.0/deltaTheta
            a2b = self.Nb_[2]
            c2b = 4.0/(deltaTheta**2)
            
            b2e = self.dNe_[2]*2.0/deltaTheta
            a2e = self.Ne_[2]
            c2e = 4.0/(deltaTheta**2)
            for i in range(1,self.nHarmonic+1):
                sinb = math.sin(i*thetaBeg)/i
                sine = math.sin(i*thetaEnd)/i
                cosb = math.cos(i*thetaBeg)/i
                cose = math.cos(i*thetaEnd)/i              
                self.harmonicIntSin[0,i-1] = cose*(c0e/(i*i)-a0e) + sine*b0e/i
                self.harmonicIntSin[0,i-1] -= cosb*(c0b/(i*i)-a0b) + sinb*b0b/i
                self.harmonicIntCos[0,i-1] = sine*(a0e-c0e/(i*i)) + cose*b0e/i
                self.harmonicIntCos[0,i-1] -= sinb*(a0b-c0b/(i*i)) + cosb*b0b/i
                
                self.harmonicIntSin[1,i-1] = cose*(c1e/(i*i)-a1e) + sine*b1e/i
                self.harmonicIntSin[1,i-1] -= cosb*(c1b/(i*i)-a1b) + sinb*b1b/i
                self.harmonicIntCos[1,i-1] = sine*(a1e-c1e/(i*i)) + cose*b1e/i
                self.harmonicIntCos[1,i-1] -= sinb*(a1b-c1b/(i*i)) + cosb*b1b/i
                
                self.harmonicIntSin[2,i-1] = cose*(c2e/(i*i)-a2e) + sine*b2e/i
                self.harmonicIntSin[2,i-1] -= cosb*(c2b/(i*i)-a2b) + sinb*b2b/i
                self.harmonicIntCos[2,i-1] = sine*(a2e-c2e/(i*i)) + cose*b2e/i
                self.harmonicIntCos[2,i-1] -= sinb*(a2b-c2b/(i*i)) + cosb*b2b/i
                
    def refreshMatrices(self, data):
        kGlob = data.getKtL()
        for inod in range(self.Nnod):
            for jh in range(self.nHarmonic):
                id1 = self.Nodes[inod].ID[0]
                if id1 >= 0:
                    kGlob[id1,data.mesh.harmonicsIDa[jh]] = 0.0
                    kGlob[id1,data.mesh.harmonicsIDb[jh]] = 0.0
                    kGlob[id1,data.mesh.harmonicsIDc[jh]] = 0.0
                    kGlob[id1,data.mesh.harmonicsIDd[jh]] = 0.0
                    kGlob[data.mesh.harmonicsIDa[jh],id1] = 0.0
                    kGlob[data.mesh.harmonicsIDb[jh],id1] = 0.0
                    kGlob[data.mesh.harmonicsIDc[jh],id1] = 0.0
                    kGlob[data.mesh.harmonicsIDd[jh],id1] = 0.0
            
    def calculate(self, data, linear = True):
        """
        Calculate matrices and vectors
        Input:
            data: a class that have methods return global matrices and vectors
            linear: True: calculate only linear part of matrices
                    False: ignore linear part and calculate nonlinear part
        """
        if not linear:
            return
        kGlob = data.getKtL()
            
        if self.current:
            self.calculateIntegrals(self.mechdof,data.deltaT*(data.istep-1))
            self.refreshMatrices(data)
        else:
            self.calculateIntegrals(self.mechdof)
            self.refreshMatrices(data)
        K = np.zeros(self.Ndof+self.nHarmonic*4,self.dtype)
        if self.normv[0] > 0.9:
            # inside layer
            for inod in range(self.Nnod):
                for jh in range(self.nHarmonic):
                    K[0] = -(self.rRatio**(jh+1))*self.harmonicIntCos[inod,jh]
                    K[1] = -(self.rRatio**(jh+1))*self.harmonicIntSin[inod,jh]
                    K[2] = self.harmonicIntCos[inod,jh]
                    K[3] = self.harmonicIntSin[inod,jh]
#                    K[0] = -self.harmonicIntCos[inod,jh]
#                    K[1] = -self.harmonicIntSin[inod,jh]
#                    K[2] = (self.rRatio**(jh+1))*self.harmonicIntCos[inod,jh]
#                    K[3] = (self.rRatio**(jh+1))*self.harmonicIntSin[inod,jh]
                    K *= self.matConstant*(jh+1)
#                    K*=-1.0
                    id1 = self.Nodes[inod].ID[0]
                    if id1 >= 0:
                        kGlob[id1,data.mesh.harmonicsIDa[jh]] += K[0]
                        kGlob[id1,data.mesh.harmonicsIDb[jh]] += K[1]
                        kGlob[id1,data.mesh.harmonicsIDc[jh]] += K[2]
                        kGlob[id1,data.mesh.harmonicsIDd[jh]] += K[3]
                        kGlob[data.mesh.harmonicsIDa[jh],id1] += K[0]
                        kGlob[data.mesh.harmonicsIDb[jh],id1] += K[1]
                        kGlob[data.mesh.harmonicsIDc[jh],id1] += K[2]
                        kGlob[data.mesh.harmonicsIDd[jh],id1] += K[3]
                        
        elif self.normv[0] < -0.9:
            # outside layer
            for inod in range(self.Nnod):
                for jh in range(self.nHarmonic):
                    K[0] = self.harmonicIntCos[inod,jh]
                    K[1] = self.harmonicIntSin[inod,jh]
                    K[2] = -(self.rRatio**(jh+1))*self.harmonicIntCos[inod,jh]
                    K[3] = -(self.rRatio**(jh+1))*self.harmonicIntSin[inod,jh]
                    K *= self.matConstant*(jh+1)
#                    K*=-1.0
                    id1 = self.Nodes[inod].ID[0]
                    if id1 >= 0:
                        kGlob[id1,data.mesh.harmonicsIDa[jh]] += K[0]
                        kGlob[id1,data.mesh.harmonicsIDb[jh]] += K[1]
                        kGlob[id1,data.mesh.harmonicsIDc[jh]] += K[2]
                        kGlob[id1,data.mesh.harmonicsIDd[jh]] += K[3]
                        kGlob[data.mesh.harmonicsIDa[jh],id1] += K[0]
                        kGlob[data.mesh.harmonicsIDb[jh],id1] += K[1]
                        kGlob[data.mesh.harmonicsIDc[jh],id1] += K[2]
                        kGlob[data.mesh.harmonicsIDd[jh],id1] += K[3]
        
            
class PolarBoundary(FB.StandardBoundary,PolarElement):
    """
    Standard Boundary Element in Polar Coordinates
    """
    def calculateGreen(self, x, xp):
        if np.allclose(x,xp,rtol=1.0e-13):
            raise FB.SingularPoint
        m = x[0]+xp[0]-2.0*x[0]*xp[0]*math.cos(x[1]-xp[1])
        self.G = math.log(m)/(4.0*math.pi)
        self.gradG[0] = (x[0]-xp[0]*math.cos(x[1]-xp[1]))/(m*2.0*math.pi)
        self.gradG[1] = xp[0]*math.sin(x[1]-xp[1])/(m*2.0*math.pi)
        
    def subCalculateKLinear(self, K, element, i, j):
        wfac = self.getFactor()
        wfact = element.getFactorX(element.detJ)
        wfacx = element.getFactorXR(element.detJ)
        K[1,0] = self.N_[i]*element.Nx_[j]*\
        (element.normv[0]*self.gradG[0]+element.normv[1]*self.gradG[1])
        K[1,0] *= wfac*wfacx
        K[1,0] += self.N_[i]*element.Nx_[j]*self.G*\
        element.normv[0]*wfac*wfact/element.xx_[0]
        K[1,1] = self.N_[i]*element.Nx_[j]*self.G
        K[1,1] *= wfac*wfact
        


def reduce_2pi(x):
    """
    get remainder of division by n times 2*pi
    """
    pi2 = np.pi*2.0
    if x < 0.0:
        res = x
        while res < 0.0:
            res += pi2
        return res
    if x > 0.0:
        res = x
        while res > pi2:
            res -= pi2
        return res
    return x
        