# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:40:01 2017

@author: haiau
"""

import numpy as np
import pylab as pl
import math
import sympy as syp
#import scipy.linalg as la
import Math.injectionArray as ia

class CommonData(object):
    """
    Common array that can be shared between elements of the same type
    This help reducing memory as well as improve speed
    """
    def __init__(self, Nnod, ndim, intData, dtype = 'float64'):
        """
        Initialize Common Data object
        Input:
            Nnod: number of node of one element
            ndim: number of dimensions
            intData: integration data
            dtype: datatype of float number
        """
        self.dtype = dtype
        self.intData = intData
        self.Nnod = Nnod
        self.Ndim = ndim
        if intData is not None:
            self.N_ = []
            self.ng = intData.getNumberPoint()
            for i in range(self.ng):
                self.N_.append(np.zeros(self.Nnod))
           
            
    def getN(self):
        try:
            return self.N_
        except AttributeError:
            return None

class Element(object):
    """
    Element class includes all properties and methods need for an Element in FEM
    This class is for Node based Element. However, it is possible to derive an
    Edge based element class from this class.    
    """
    def __init__(self, Nodes, pd, basisFunc, material, intData,\
    dtype = 'float64', commonData = None):
        """
        Initialize an Element (node based)
        Input:
            Nodes: list of nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunc: function that takes x as input and outputs
                an array N of shape functions and their derivatives dN
                This function must have N_ and dN_ as optional parameters for
                update N_ and dN_
            material: material of element
            intData: integration data
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
        """
        self.dtype = dtype
        self.Nodes = Nodes
        self.Nnod = len(Nodes)
        self.pd = pd
        if self.Nnod >= 1:
            self.Ndim = Nodes[0].getNdim()
            self.Ndof = Nodes[0].getNdof()
            self.timeOrder= Nodes[0].getTimeOrder()
        else:
            raise ElementNoNode

        self.basisFunc = basisFunc
        self.material = material
        self.intData = intData
        
        if commonData is None:
            self.commonData = CommonData(self.Nnod,self.Ndim,intData,dtype)
        else:
            self.commonData = commonData
        if commonData is not None and commonData.intData != intData:
            assert 'incompatible between common data and integration data'
        try:
            self.Ns_ = self.commonData.getN()
        except:
            self.Ns_ = None
        self.dNs_ = None
        self.N_ = None
        self.dN_ = None
        self.x_ = np.zeros(self.Ndim,dtype)
        self.u_ = np.zeros(self.Ndof,dtype)
        self.v_ = np.zeros(self.Ndof,dtype)
        self.a_ = np.zeros(self.Ndof,dtype)
        self.gradu_ = np.zeros((self.Ndim,self.Ndof),dtype)
        self.__temp_grad_u = np.zeros((self.Ndim,self.Ndof),dtype)
        self.__temp_u = np.zeros(self.Ndof,dtype)
        self.bodyLoad = None
        self.linear = False
        self.calculateBoundingBox()
        self.current = False
        self.movingVel = None
        self.expr = [] # sympy expressions and corresponding parameters
        self.lmdfy = []# lambdified version of expr
        
    def __str__(self):
        """
        Print information about element
        """
        s = 'Name of element: '+self.__class__.__name__
        s += '\n'
        s += str(self.Ndim) + '-dimensional Element\n'
        s += 'Number of nodes: '+str(self.Nnod)
        s += '\n'
        s += 'Basis function degree(s): '+ str(self.pd)
        s += '\n'
        if not self.basisFunc is None:
            s += 'Basis function: '+self.basisFunc.__name__
        s += '\n'
        s += 'Material: '+self.material.__class__.__name__
        s += '\n'
        s += 'Info of each node:\n'
        for i in range(self.Nnod):
            s += 'Node '+str(i)+'--------------------\n'
            s += str(self.Nodes[i])
            
        return s
        
    def __contains__(self, item):
        return item in self.Nodes
    
    def addExpression(self, expr, parm):
        exprx = syp.sympify(expr)
        parmx = (syp.sympify(p) for p in parm)
        self.expr.append((exprx,parmx))
        self.lmdfy.append(syp.lambdify(parmx,exprx,'numpy'))
        
    def setMovingVelocity(self, v):
        """
        set prescribed moving velocity
        v: array, velocities
        """
        self.current = True
        self.movingVel = np.array(v,self.dtype)
        
    def isLinear(self):
        """
        Return true if the element is linear
        false otherwise
        """
        return self.linear
        
    def calculateBoundingBox(self):
        """
        Calculate bounding box of element
        """
        x = [n.getX()[0] for n in self.Nodes]
        y = [n.getX()[1] for n in self.Nodes]
        self.__maxx__ = max(x)
        self.__minx__ = min(x)
        self.__maxy__ = max(y)
        self.__miny__ = min(y)
        
    def getBoundingBox(self):
        """
        Return bounding box of element
        """
        return self.__minx__, self.__maxx__, self.__miny__, self.__maxy__
        
    def setLinearity(self, lin):
        """
        Set linearity for this element
        If lin is set to be True, the matrices and vectors will not
        be calculated in iterations steps. They will be calculated if False is
        specified
        """
        self.linear = lin
        
    def hasNodes(self, nodes):
        """
        return True if nodes belongs to this element
        false otherwise
        """
        try:
            for n in nodes:
                if n not in self.Nodes:
                    return False
            return True
        except TypeError:
            return False
        
    def getNodes(self):
        """
        return node list
        """
        return self.Nodes
        
    def getNnod(self):
        """
        return number of nodes
        """
        return self.Nnod
        
    def getNdim(self):
        """
        return number of dimensions
        """
        return self.Ndim
        
    def calculateBasis(self, x_, nodeOrder):
        """
        Calculate and initialize an array of basis functions at x
        Input:
            x_ : parametric coordinate
        """
        if not self.N_ is None:
            self.basisFunc(x_, self.pd, self.N_, self.dN_, nodeOrder)
        else:
            self.N_, self.dN_ = self.basisFunc(x_, self.pd, Order = nodeOrder)
            if self.N_.size != self.Nnod:
                raise ElementBasisFunctionNodeMismacht
        
    def getN(self):
        """
        Return shape functions
        """
        return self.N_
        
    def getDN(self):
        """
        Return derivatives of shape functions
        """
        return self.dN_
            
    def getX(self, x, N_ = None):
        """
        Get coordinates at parametric coordinates x (x from calculateBasis)
        """
        if N_ is None:
            N_ = self.N_
        
        try:
            for i in range(self.Nnod):
                x += self.Nodes[i].getX()*N_[i]
        except ValueError:
            print('here')
        except TypeError:
            raise ElementBasisFunctionNotCalculated
        
    def getU(self, u, N_ = None):
        """
        Get displacement at parametric coordinates x (x from calculateBasis)
        Input:
            u : mutable result
            data: class that can return global displacement vector(deprecated)
        """
        if N_ is None:
            N_ = self.N_
            
        try:
            for i in range(self.Nnod):
#                u += self.Nodes[i].getFromGlobalU(data.getU(),self.__temp_u)*\
#                self.N_[i]
                u += self.Nodes[i].getU()*N_[i]
        except TypeError:
            raise ElementBasisFunctionNotCalculated
        
    def getGradU(self, gradu, data, dN_ = None):
        """
        Get gradient of displacement
        Input:
            u : mutable result
            data: class that can return global displacement vector
        """
        if dN_ is None:
            dN_ = self.dN_
        #gradu.fill(0.0)
        try:
            for i in range(self.Nnod):
                np.outer(dN_[:,i:i+1],\
                self.Nodes[i].getFromGlobalU(data.getU(),self.__temp_u),\
                self.__temp_grad_u)
                gradu += self.__temp_grad_u
        except TypeError:
            raise ElementBasisFunctionNotCalculated
            
    def getGradUP(self, gradu, dN_ = None):
        if dN_ is None:
            dN_ = self.dN_
            
        gradu.fill(0.0)
        for i in range(self.Nnod):
            np.outer(dN_[:,i:i+1],\
            self.Nodes[i].getU().toNumpy(),self.__temp_grad_u)
            gradu += self.__temp_grad_u
    
    def getV(self, u, N_ = None):
        """
        Get velocity at parametric coordinates x (x from calculateBasis)
        Input:
            u : mutable result
            data: class that can return global velocity vector(deprecated)
        """
#        if data.getTimeOrder() < 1:
#            raise TimeOrderMismatch
        if N_ is None:
            N_ = self.N_    
        try:
            for i in range(self.Nnod):
#                u += self.Nodes[i].getFromGlobalV(data.getV(),self.__temp_u)*\
#                self.N_[i]
                u += self.Nodes[i].getV()*N_[i]
        except TypeError:
            raise ElementBasisFunctionNotCalculated

        
    def getA(self, u, N_ = None):
        """
        Get acceleration at parametric coordinates x (x from calculateBasis)
        Input:
            u : mutable result
            data: class that can return global acceleration vector(deprecated)
        """
        if N_ is None:
            N_ = self.N_
#        if data.getTimeOrder() < 2:
#            raise TimeOrderMismatch
        try:
            for i in range(self.Nnod):
#                u += self.Nodes[i].getFromGlobalA(data.getA(),self.__temp_u)*\
#                self.N_[i]
                u += self.Nodes[i].getA()*N_[i]
        except TypeError:
            raise ElementBasisFunctionNotCalculated
            
    def getAllValues(self, data, current = False, mechDofs = None):
        """
        Get all the current values: displacement, velocity and acceleration
        Input:
            data: a class that have methods return global vectors
            current: current configuration
            mechDofs: an array contains all mechanical dof indices
        """
        # Calculate coordinate at current Gaussian point
        self.x_.fill(0.0)
        self.getX(self.x_)
        
        # calculate displacement
        self.u_.fill(0.0)            
        self.getU(self.u_)
        
        if current:
            if not mechDofs is None:
                for i in range(len(mechDofs)):
                    self.x_[i] += self.u_[mechDofs[i]]
            if self.movingVel is not None:
                for i in range(len(self.movingVel)):
                    self.x_[i] += self.movingVel[i]*data.deltaT
        
        # calculate gradient of displacement
        self.gradu_.fill(0.0)
        try:
            self.getGradU(self.gradu_,data)
        except AttributeError:
            self.getGradUP(self.gradu_)
        
        # calculate velocity
        if self.timeOrder > 0:
            self.v_.fill(0.0)
            self.getV(self.v_)
            
        # calculate acceleration
        if self.timeOrder == 2:
            self.a_.fill(0.0)
            self.getA(self.a_)
            
    def updateValue(self, uGlobal, vGlobal = None, aGlobal = None):
        """
        update the current value of displacement, velocity and acceleration
        """
        for i in range(self.Nnod):
            n = self.Nodes[i]
            n.updateU(uGlobal)
            n.updateV(vGlobal)
            n.updateA(aGlobal)

    def setBodyLoad(self, loadfunc, id_b = 0):
        """
        set body load function
        """
        if loadfunc is None:
            return
        assert callable(loadfunc),\
        'Body Load must be a function of position and time'
        self.bodyLoad = loadfunc
        self.id_bodyLoad = id_b
        
    def getBodyLoad(self, t):
        """
        get the body load
        """
        if self.bodyLoad is not None:
            try:
                return self.bodyLoad(self.x_,t)
            except TypeError:
                print(self.bodyLoad)
        else:
            return 0.0
            
    def hasBodyLoad(self):
        """
        Check if element has body load
        """
        try:
            return (self.bodyLoad is not None) or ('calculateRe' in dir(self))
        except AttributeError:
            return False
            
    def setMaterial(self, mat):
        """
        Set material of element
        """
        self.material = mat
        
    def getMaterial(self):
        return self.material
        
    def calculate(self, data):
        """
        Calculate element matrices and vectors
        This method will be implemented in derived classes
        Input:
            data: a class that have methods return global matrices and vectors
        """
        pass
    

# End of Element class definition

class StandardElement(Element):
    """
    Standard Quadrilateral Element
    This is super class for further derivations
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, material, intData,\
    dtype = 'float64', commonData = None, ndime = 2):
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
            ndime: 2: 2-d element
                   3: 3-d element
                   1: 1-d element
        """
        Element.__init__(self, Nodes, pd, basisFunction, material, intData,\
        dtype, commonData)
        self.Ndime = ndime
        self.nodeOrder = nodeOrder
        # Temporary variables
        self.ig = 0
        self.factor = [0.0]*self.intData.getNumberPoint()
        self.dNs_ = []
        for i in range(self.intData.getNumberPoint()):
            self.dNs_.append(np.zeros((self.Ndim,self.Nnod)))
        self.calculateBasis(self.nodeOrder)
        self.__prepare_matrices__()
#        self.R = ia.zeros(self.Ndof*self.Nnod,self.dtype)
#        self.K = np.zeros((self.Ndof,self.Ndof),self.dtype)
#        self.D = None
#        self.M = None
#        if self.Nodes[0].getTimeOrder() > 0:
#            self.D = ia.zeros((self.Ndof,self.Ndof),self.dtype)
#        
#        if self.Nodes[0].getTimeOrder() == 2:
#            self.M = ia.zeros((self.Ndof,self.Ndof),self.dtype)
    
    def __prepare_matrices__(self):
        self.K = []
        self.R = []
        self.D = None
        self.M = None
        if self.timeOrder > 0:
            self.D = []
        if self.timeOrder == 2:
            self.M = []
        for i in range(self.Nnod):
            self.K.append([])
            self.R.append(ia.zeros(self.Ndof,self.dtype))
            if self.timeOrder > 0:
                self.D.append([])
            if self.timeOrder == 2:
                self.M.append([])
            for j in range(self.Nnod):
                self.K[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
                if self.timeOrder > 0:
                    self.D[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
                if self.timeOrder == 2:
                    self.M[i].append(ia.zeros((self.Ndof,self.Ndof),self.dtype))
    
    def connect(self, *arg):
        """
        Connect local matrices to global matrices
        """        
        rGlob = arg[0]
        rGlobD = arg[1]
        kGlob = arg[2]
        kGlobD = arg[3]
        dGlob = arg[4]
        dGlobD = arg[5]
        mGlob = arg[6]
        mGlobD = arg[7]
        for inod in range(self.Nnod):
            ids1 = self.Nodes[inod].getID()
            for idof in range(self.Ndof):
                id1 = ids1[idof]
                if id1 >= 0:
                    self.R[inod].connect(idof,rGlob,id1)
                elif id1 < -1:
                    self.R[inod].connect(idof,rGlobD,-id1-2)
            for jnod in range(self.Nnod):
                ids2 = self.Nodes[jnod].getID()
                for idof in range(self.Ndof):
                    id1 = ids1[idof]
                    if id1 >= 0:
                        for jdof in range(self.Ndof):
                            id2 = ids2[jdof]
                            if id2 >= 0:
                                self.K[inod][jnod].connect((idof,jdof),kGlob,\
                                (id1,id2))
                                if self.timeOrder > 0:
                                    self.D[inod][jnod].connect((idof,jdof),\
                                    dGlob,(id1,id2))
                                if self.timeOrder == 2:
                                    self.M[inod][jnod].connect((idof,jdof),\
                                    mGlob,(id1,id2))
                            elif id2 < -1:
                                self.K[inod][jnod].connect((idof,jdof),kGlobD,\
                                (id1,-id2-2))
                                if self.timeOrder > 0:
                                    self.D[inod][jnod].connect((idof,jdof),\
                                    dGlobD,(id1,-id2-2))
                                if self.timeOrder == 2:
                                    self.M[inod][jnod].connect((idof,jdof),\
                                    mGlobD,(id1,-id2-2))

        
    
    def getFactor(self):
        return self.factor[self.ig]
        
    def Jacobian(self, dN_, coord = 'r', edg = False):
        """
        Calculate det(J) and modify dN_
        coord = 'r' : Cartesian coordinates (default) (deprecated)
                'c' : cylindrical coordinates x[0] = r, x[1] = phi, x[2] = z
                's' : sphercial coordinates x[0] = r, x[1] = phi, x[2] = theta
        return det(J)
        """
        __tempJ__ = np.zeros((self.Ndim,self.Ndim),self.dtype)
        
        Jmat = np.zeros((self.Ndim,self.Ndim),self.dtype)
#        if coord == 'r':
        for i in range(self.Nnod):
            #np.outer(self.Nodes[i].getX(),dN_[:,i],temp)
            np.outer(dN_[:,i],self.Nodes[i].getX(),__tempJ__)
            Jmat += __tempJ__
#        elif coord == 'c':
#            for i in range(self.Nnod):
#                x = self.Nodes[i].getX()
#                x[1] *= x[0]
#                np.outer(dN_[:,i],x,__tempJ__)
#                Jmat += __tempJ__
#        elif coord == 's':
#            for i in range(self.Nnod):
#                x = self.Nodes[i].getX()
#                x[1] *= x[0]
#                try:
#                    x[2] *= x[0]*math.sin(x[1])
#                except:
#                    pass
#                np.outer(dN_[:,i],x,__tempJ__)
#                Jmat += __tempJ__
#        else:
#            raise ValueError
        
        if edg:
            self.detEdg1 = math.sqrt(Jmat[0,0]**2+Jmat[1,0]**2)
            self.detEdg2 = math.sqrt(Jmat[0,1]**2+Jmat[1,1]**2)
        
        det = __determinant__(Jmat)
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
        
    def calculateBasis(self, NodeOrder):
        ig = 0
        for xg, wg in self.intData:
            self.basisND(xg, self.Ns_[ig], self.dNs_[ig])
            self.factor[ig] = self.Jacobian(self.dNs_[ig])*np.prod(wg)
            ig += 1
            
    def prepareElement(self):
        self.calculateBasis(self.nodeOrder)
            
    def getBasis(self):
        self.dN_ = self.dNs_[self.ig]
        self.N_ = self.Ns_[self.ig]
        
    def basisND(self, x_, N_, dN_):
        """
        n-dimensional basis functions, n = 1, n = 2 or n = 3
        n = self.Ndim
        """ 
        if self.Ndim == 1:
            return self.basisFunc(x_,self.pd,N_,dN_)
        
        try:
            n1 = self.pd[0] + 1
        except TypeError:
            if self.Ndime == 1:
                try:
                    return self.basisFunc(x_,self.pd,N_,dN_[0,:])
                except IndexError:
                    N_.fill(0.0)
                    dN_.fill(0.0)
                    return self.basisFunc(x_,self.pd,N_,dN_)
                
            
        n2 = self.pd[1] + 1
        n3 = 1
        if self.Ndim == 3:
            n3 = self.pd[2] + 1
        sz = n1*n2*n3
        if N_ is None:            
            N_ = np.empty(sz,self.dtype)
        if dN_ is None:
            dN_ = np.empty((self.Ndim,sz),self.dtype)
            
        Order = self.nodeOrder
            
        N1,dN1 = self.basisFunc(x_[0],self.pd[0])
        N2,dN2 = self.basisFunc(x_[1],self.pd[1])
        if self.Ndim == 3:
            N3,dN3=self.basisFunc(x_[2],self.pd[2])
        for i in range(sz):
            N_[i] = N1[Order[0][i]]*N2[Order[1][i]]
            dN_[0,i]= dN1[Order[0][i]]*N2[Order[1][i]]
            dN_[1,i]= N1[Order[0][i]]*dN2[Order[1][i]]
            if self.Ndim == 3:
                N_[i] *= N3[Order[2][i]]
                dN_[0,i] *= N3[Order[2][i]]
                dN_[1,i] *= N3[Order[2][i]]
                dN_[2,i] = N1[Order[0][i]]*\
                N2[Order[1][i]]*dN3[Order[2][i]]
            
    
        return N_, dN_    
        
    def getNodeOrder(self):
        """
        Return node ordering of element
        """
        return self.nodeOrder
        
    def postCalculate(self, x, val = 'u'):
        """
        Return u, v, a, gradu at position x (in natural coordinates)
        """
        if np.any(np.fabs(x)>1.0+1.0e-13):
            raise OutsideElement
        #N_ = np.zeros(self.Nnod,self.dtype)
        #dN_ = np.zeros((self.Ndim,self.Nnod),self.dtype)
        N_ = self.Ns_[0]
        dN_ = self.dNs_[0]
        self.basisND(x,N_,dN_)
        self.Jacobian(dN_)
        if val == 'u':
            #u_ = np.zeros(self.Ndof,self.dtype)
            self.u_.fill(0.0)
            self.getU(self.u_,N_)
            return self.u_
        if val == 'gradu':
            #gradu = np.zeros(self.Ndof,self.dtype)
            self.gradu_.fill(0.0)
            self.getGradUP(self.gradu_,dN_)
            return self.gradu_
        if val == 'v':
            if self.timeOrder > 0:
                #v_ = np.zeros(self.Ndof,self.dtype)
                self.v_.fill(0.0)
                self.getV(self.v_,N_)
                return self.v_
            else:
                raise Exception('No velocity')
        if val == 'a':
            if self.timeOrder > 0:
                #a_ = np.zeros(self.Ndof,self.dtype)
                self.a_.fill(0.0)
                self.getV(self.a_,N_)
                return self.a_
            else:
                raise Exception('No acceleration')
                
    def insideBoundingBox(self, x):
        return x[0]<self.__maxx__+1.0e-13 and x[0]>self.__minx__-1.0e-13 \
        and x[1]<self.__maxy__+1.0e-13 and x[1]>self.__miny__-1.0e-13
                
    def getValFromX(self, x, val = 'u'):
        """
        Return natural coordinate xi corresponding to physical coordinate x
        Raise OutsideEdlement if x is not inside element.
        """
        if not self.insideBoundingBox(x):
            raise OutsideElement
            
        matA = np.ones((self.Ndim+1,self.Nnod),self.dtype)
        vecb = np.ones(self.Ndim+1,self.dtype)
        if self.Ndim == 1:
            for i in range(self.Nnod):
                matA[1,i] = self.Nodes[i].getX()
            vecb[1] = x
        if self.Ndim == 2:
            for i in range(self.Nnod):
                x_ = self.Nodes[i].getX()
                matA[1,i] = x_[0]
                matA[2,i] = x_[1]
            vecb[1] = x[0]
            vecb[2] = x[1]
        if self.Ndim == 3:
            for i in range(self.Nnod):
                x_ = self.Nodes[i].getX()
                matA[1,i] = x_[0]
                matA[2,i] = x_[1]
                matA[3,i] = x_[2]
            vecb[1] = x[0]
            vecb[2] = x[1]
            vecb[3] = x[2]
            
        N_,_,_,_ = np.linalg.lstsq(matA,vecb)
        if val == 'u':
            #u_ = np.zeros(self.Ndof,self.dtype)
            self.getU(self.u_,N_)
            return self.u_
        if val == 'v':
            if self.timeOrder > 0:
                #v_ = np.zeros(self.Ndof,self.dtype)
                self.getV(self.v_,N_)
                return self.v_
            else:
                raise Exception('No velocity')
        if val == 'a':
            if self.timeOrder > 0:
                #a_ = np.zeros(self.Ndof,self.dtype)
                self.getV(self.a_,N_)
                return self.a_
            else:
                raise Exception('No acceleration')    
        
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
            raise OutsideElement
            
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
                np.outer(self.Nodes[i].getX(),dN_[:,i],__tempJ__)
                #np.outer(dN_[:,i],self.Nodes[i].getX(),__tempJ__)
                Jmat += __tempJ__
            if np.allclose(__determinant__(Jmat),0.0,rtol=1.0e-14):
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
                    raise OutsideElement
                else:
                    return xi
        raise OutsideElement
        
        
        
    def initializeMatrices(self):
        self.R.fill(0.0)
        self.K.fill(0.0)
        if not self.D is None:
            self.D.fill(0.0)
        if not self.M is None:
            self.M.fill(0.0)
            
    def calculateBodyLoad(self, data):
        """
        Calculate body load
        """
        if not self.hasBodyLoad():
            return
        vGlob = data.getRe()
        vGlobD = None
#        vGlobD = data.getRiD()
        
        GaussPoints = self.intData
        R = np.zeros(self.Ndof,self.dtype)
        self.ig = -1
        try:
            t = data.getTime()
        except AttributeError:
            t = 0.0
            pass
            
        # loop over Gaussian points
        for xg, wg in GaussPoints:
            self.ig += 1
            self.getBasis()
            self.getAllValues(data)
            try:
                self.calculateFES()
            except AttributeError:
                pass
            self.material.calculate(self)
            for i in range(self.Nnod):
                try:
                    self.calculateRe(R,i,t)
                    assembleVector(vGlob, vGlobD, R, self.Nodes[i])
                except AttributeError:
                    pass
                
    def integrate(self, intFunc, res, intData = None, edg = False):
        """
        Integrate intFunc over element
        store result in res
        return res
        """
        if intData is None:
            GaussPoints = self.intData
        
            for self.ig in range(GaussPoints.Npoint):
                self.getBasis()
                self.getAllValues(None)
                self.material.calculate(self)
                res += intFunc(self)
            return res
        else:
            if edg:
                edgn = intData.edg
            N = np.zeros(self.Nnod,self.dtype)
            dN = np.zeros((self.Ndim,self.Nnod),self.dtype)
            for xg,wg in intData:
                self.basisND(xg,N,dN)
                detJ = self.Jacobian(dN,edg)
                if edg:
                    if edgn == 2 or edgn == 4:
                        wei = self.detEdg2*np.prod(wg)
                    else:
                        wei = self.detEdg1*np.prod(wg)
                else:
                    wei = detJ*np.prod(wg)
                self.x_.fill(0.0)
                self.getX(self.x_,N)
                self.u_.fill(0.0)
                self.getU(self.u_,N)
                if self.timeOrder > 0:
                    self.v_.fill(0.0)
                    self.getV(self.v_,N)
                if self.timeOrder == 2:
                    self.a_.fill(0.0)
                    self.getA(self.a_,N)
                self.gradu_.fill(0.0)
                self.getGradUP(self.gradu_,dN)
                self.material.calculate(self)
                res += intFunc(self)*wei
            return res
    
    def calculate(self, data, linear = False):
        """
        Calculate matrices and vectors
        Input:
            data: a class that have methods return global matrices and vectors
            linear: True: calculate only linear part of matrices
                    False: ignore linear part and calculate nonlinear part
        """
        # Get matrices from data
        timeOrder = data.getTimeOrder()
        try:
            t = data.getTime()
        except AttributeError:
            t = 0.0
        if linear:
            vGlob = data.getRiL()
            vGlobD = data.getRiLD()
            kGlob = data.getKtL()
            kGlobD = data.getKtLd()
            dGlob = data.getDL()
            dGlobD = data.getDLd()
            mGlob = data.getML()
            mGlobD = data.getMLd()
        
        #Gauss integration points
        GaussPoints = self.intData
        
        # initialized matrices and vector
        if not linear:
            R = self.R
            K = self.K
            D = self.D
            M = self.M
        else:
            R = np.zeros(self.Ndof,self.dtype)
            K = np.zeros((self.Ndof,self.Ndof),self.dtype)
            if self.timeOrder > 0:
                D = np.zeros((self.Ndof,self.Ndof),self.dtype)
            else:
                D = None
            if self.timeOrder == 2:
                M = np.zeros((self.Ndof,self.Ndof),self.dtype)
            else:
                M = None
                
        self.ig = -1
            
        # loop over Gaussian points
        for xg, wg in GaussPoints:
            self.ig += 1
#            self.calculateBasis(xg, self.nodeOrder)
#            self.factor = self.Jacobian(self.dN_)*np.prod(wg)
            self.getBasis()
            
            # Calculate coordinate at current Gaussian point
            self.getAllValues(data)
            try:
                self.calculateFES()
            except AttributeError:
                pass
            
            # Initialize matrices
            #self.initializeMatrices()
            
            self.material.calculate(self)
            
            if linear:
                for i in range(self.Nnod):
                    # calculate and assemble load vector
                    try:
                        self.calculateRLinear(R,i,t)
                        assembleVector(vGlob, vGlobD, R, self.Nodes[i])
                    except AttributeError:
                        pass
                    
                    # loop over node j
                    try:
                        for j in range(self.Nnod):
                            # calculate and assemble matrices
                            self.calculateKLinear(K,i,j,t)
                            assembleMatrix(kGlob,kGlobD,K,\
                            self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                    try:
                        if timeOrder > 0:
                            for j in range(self.Nnod):
                                self.calculateDLinear(D,i,j,t)
                                assembleMatrix(dGlob,dGlobD,D,\
                                self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                    try:                        
                        if timeOrder == 2:
                            for j in range(self.Nnod):
                                self.calculateMLinear(M,i,j,t)
                                assembleMatrix(mGlob,mGlobD,M,\
                                self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                continue
            # loop over node i            
            for i in range(self.Nnod):
                # calculate and assemble load vector
                try:
                    self.calculateR(R[i],i,t)
                except AttributeError:
                    pass
                
                #assembleVector(vGlob, vGlobD, R, self.Nodes[i])
                
                # loop over node j
                try:
                    self.calculateK(K[i][0],i,0,t)
                    #assembleMatrix(kGlob,kGlobD,K,\
                    #self.Nodes[i],self.Nodes[0])
                    for j in range(1,self.Nnod):
                        # calculate and assemble matrices
                        self.calculateK(K[i][j],i,j,t)
                        #assembleMatrix(kGlob,kGlobD,K,\
                        #self.Nodes[i],self.Nodes[j])
                except AttributeError:
                    pass
                try:
                    self.calculateD(D[i][0],i,0,t)
                    #assembleMatrix(dGlob,dGlobD,D,\
                    #self.Nodes[i],self.Nodes[0])
                    for j in range(1,self.Nnod):
                        self.calculateD(D[i][j],i,j,t)
                        #assembleMatrix(dGlob,dGlobD,D,\
                        #self.Nodes[i],self.Nodes[j])
                except (AttributeError, TypeError):
                    pass
                try:
                    self.calculateM(M[i][0],i,0,t)
                    #assembleMatrix(mGlob,mGlobD,M,\
                    #self.Nodes[i],self.Nodes[0])                        
                    for j in range(1,self.Nnod):
                        self.calculateM(M[i][j],i,j,t)
                        #assembleMatrix(mGlob,mGlobD,M,\
                        #self.Nodes[i],self.Nodes[j])
                except (AttributeError, TypeError):
                    pass
                
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None,\
             deformed = False, deformed_factor=1.0 ):
        """
        Default plot method
        This method simply plot the nodes continuously
        Derived classes should override this method to get desired shape
        deformed: plot deformed structure, the DOFs of deformations are always
        from 0:Ndim
        """
        d_fact = deformed_factor
        if fig is None:
            fig = pl.figure()
        if not deformed:
            x = [n.getX()[0] for n in self.Nodes]
        else:
            x = [n.getX()[0] + d_fact*n.getU()[0] for n in self.Nodes]
        if self.Ndim > 1:
            if not deformed:
                y = [n.getX()[1] for n in self.Nodes]
            else:
                y = [n.getX()[1] + d_fact*n.getU()[1] for n in self.Nodes]
        #if self.Ndim == 3:
        #    z = [n.getX()[2] for n in self.Nodes]
        if self.Ndim == 1:
            pl.plot(np.array(x),col)
        if self.Ndim == 2:
            pl.plot(np.array(x),np.array(y),col)
            
        if self.Ndim == 3:
            """
            Will be implemented later
            """
            return None
        if number is not None:
            c = 0.5*(self.Nodes[0].getX()+self.Nodes[-1].getX())
            pl.text(c[0],c[1],str(number))
        return fig,[n.getX() for n in self.Nodes]
            

# End of class StandardElement definition

class ElementBasisFunctionNodeMismacht(Exception):
    """
    Exception in case of number of basis functions and number of node mismatcht
    """
    pass

    
class ElementBasisFunctionNotCalculated(Exception):
    """
    Exception in case of basis functions were not calculated
    """
    pass

class ElementNoNode(Exception):
    """
    Exception in case of element initialized with no node
    """
    pass

class TimeOrderMismatch(Exception):
    """
    Exception in case of time order mismatch
    """
    pass

class OutsideElement(Exception):
    """
    Exception in case of point being outside element
    """
    pass

class ElementError(Exception):
    """
    Exception for errors of element
    """
    pass

def assembleMatrix(mGlob, mGlobD, mLoc, iNode, jNode):
    """
    Assemble a local matrix to global matrix
    Input:
        mGlob: global matrix, numpy matrix
        mGlobD: gobal matrix for constrained dofs, numpy matrix
        mLoc: local matrix, numpy matrix
        iNode, jNode: Two Node objects
    """
    if mGlob is None:
        return
    iN = iNode.getNdof()
    jN = jNode.getNdof()
    iD = iNode.getID()
    jD = jNode.getID()
    for i in range(iN):
        idx = iD[i]
        if idx >= 0:
            for j in range(jN):
                jdx = jD[j]
                if jdx >= 0:
                    mGlob[idx,jdx] += mLoc[i,j]
                elif jdx < -1:
                    mGlobD[idx,-jdx-2] += mLoc[i,j]
                    

                    
def assembleVector(vGlob, vGlobD, vLoc, iNode):
    """
    Assemble a local vector to global vector
    Input:
        vGlob: global vector, numpy array
        vGlobD: gobal vector for constrained dofs, numpy array
        vLoc: local vector, numpy array
        iNode: Node objects
    """
    if vGlob is None:
        return
    iN = iNode.getNdof()
    iD = iNode.getID()
    for i in range(iN):
        idx = iD[i]
        if idx >= 0:
            vGlob[idx] += vLoc[i]
        elif idx < -1:
            vGlobD[-idx - 2] += vLoc[i]
    
def __determinant__(mat):
    if mat.shape == (2,2):
        det = mat[0,0]*mat[1,1] - mat[1,0]*mat[0,1]
        if np.allclose(det,0.0,rtol = 1.0e-14):
            return det
        a = mat[0,0]
        mat[0,0] = mat[1,1]/det
        mat[1,0] = -mat[1,0]/det
        mat[0,1] = -mat[0,1]/det
        mat[1,1] = a/det
        return det
    if mat.shape == (3,3):
        det = mat[0,0]*(mat[1,1]*mat[2,2]-mat[1,2]*mat[2,1])
        det -= mat[0,1]*(mat[1,0]*mat[2,2]-mat[2,0]*mat[1,2])
        det += mat[0,2]*(mat[1,0]*mat[2,1]-mat[2,0]*mat[1,1])
        a = mat[1,1]*mat[2,2]-mat[2,1]*mat[1,2]
        b = mat[0,2]*mat[2,1]-mat[2,2]*mat[0,1]
        c = mat[0,1]*mat[1,2]-mat[1,1]*mat[0,2]
        d = mat[1,2]*mat[2,0]-mat[2,2]*mat[1,0]
        e = mat[0,0]*mat[2,2]-mat[2,0]*mat[0,2]
        f = mat[0,2]*mat[1,0]-mat[1,2]*mat[0,0]
        g = mat[1,0]*mat[2,1]-mat[2,0]*mat[1,1]
        h = mat[0,1]*mat[2,0]-mat[2,1]*mat[0,0]
        i = mat[0,0]*mat[1,1]-mat[1,0]*mat[0,1]
        mat[0,0] = a/det
        mat[0,1] = b/det
        mat[0,2] = c/det
        mat[1,0] = d/det
        mat[1,1] = e/det
        mat[1,2] = f/det
        mat[2,0] = g/det
        mat[2,1] = h/det
        mat[2,2] = i/det
        return det
        
def find_node_from_order(inod, jnod, nodes, nodeOrder):
    """
    return the node has order [inod,jnod]
    """        
    for i in range(len(nodes)):
        if nodeOrder[0][i] == inod and nodeOrder[1][i] == jnod:
            return nodes[i],i
            
    raise ValueError    
    
def lineIntegrate(nodes, basisFunc, material, intFunc, intData,\
 res, co='cart'):
    """
    Integrate over line or edge
    Input:
          nodes: list or tuple of nodes
          basisFunc: basis function N,dN = basisFunc(xi)
          material: material parameters
          intFunc: function to be integrated 
                   res = intFunc(N,dN,x,u,v,gradu,mat,w,res)
          intData: quadrature data (normally Gaussian points)
          res: result
          co: coordinates system
              'cart': cartesian
              'polar': polar
    """
    nnod = len(nodes)
    ndim = nodes[0].Ndim
    N = np.zeros(nnod,'float64')
    dN = np.zeros(nnod,'float64')
    dN_ = np.zeros((ndim,nnod),'float64')
    if co == 'polar':
        loop = [0]*nnod
        for i in range(1,nnod):
            phi1 = nodes[i].getX(loop[i])[1]
            phi0 = nodes[i-1].getX(loop[i-1])[1]
            if phi1 < phi0:
                loop[i] = 1
        X = np.array([n.getX(loop[i]).tolist() for i,n in enumerate(nodes)])
    else:
        X = np.array([n.getX().tolist() for n in nodes])
        
    U = np.array([n.getU().tolist() for n in nodes])
    V = np.array([n.getV().tolist() for n in nodes])
    for xg,wg in intData:
        basisFunc(xg,nnod-1,N,dN)
        dX_dxi = np.dot(dN,X)
        x = np.dot(N,X)
        u = np.dot(N,U)
        v = np.dot(N,V)
        dN_.fill(0.0)
        if math.fabs(dX_dxi[0]) > 1.0e-15:
            dN_[0,:] = dN/dX_dxi[0]
        if math.fabs(dX_dxi[1]) > 1.0e-15:
            dN_[1,:] = dN/dX_dxi[1]
        gradu = np.dot(dN_,U)
        w = wg*math.sqrt(dX_dxi[0]**2 + dX_dxi[1]**2)
        res = intFunc(N,dN,x,u,v,gradu,material,w,res)
        
    return res
        
        
    
