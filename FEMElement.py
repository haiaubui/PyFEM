# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:40:01 2017

@author: haiau
"""

import numpy as np
import pylab as pl
#import scipy.linalg as la
import injectionArray as ia

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
        
    def isLinear(self):
        """
        Return true if the element is linear
        false otherwise
        """
        return self.linear
        
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
        
        # calculate gradient of displacement
        self.gradu_.fill(0.0)
        self.getGradU(self.gradu_,data)
        
        # calculate velocity
        if data.getTimeOrder() > 0:
            self.v_.fill(0.0)
            self.getV(self.v_)
            
        # calculate acceleration
        if data.getTimeOrder() == 2:
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

    def setBodyLoad(self, loadfunc):
        """
        set body load function
        """
        if loadfunc is None:
            return
        assert callable(loadfunc),\
        'Body Load must be a function of position and time'
        self.bodyLoad = loadfunc
        
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
    Quadrilateral Element
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
    
    def connect(self, rGlob, rGlobD, kGlob, kGlobD,\
    dGlob, dGlobD, mGlob, mGlobD):
        """
        Connect local matrices to global matrices
        """        
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
                            id2 = ids2[idof]
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
        
    def Jacobian(self, dN_):
        """
        Calculate det(J) and modify dN_
        return det(J)
        """
        __tempJ__ = np.zeros((self.Ndim,self.Ndim),self.dtype)
        
        Jmat = np.zeros((self.Ndim,self.Ndim),self.dtype)
        for i in range(self.Nnod):
            #np.outer(self.Nodes[i].getX(),dN_[:,i],temp)
            np.outer(dN_[:,i],self.Nodes[i].getX(),__tempJ__)
            Jmat += __tempJ__
        
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
            a = Jmat[0,0]*dN_[0,i]+Jmat[0,1]*dN_[1,i]
            b = Jmat[1,0]*dN_[0,i]+Jmat[1,1]*dN_[1,i]
            dN_[0,i] = a
            dN_[1,i] = b
        
        return det
        
    def calculateBasis(self, NodeOrder):
        ig = 0
        for xg, wg in self.intData:
            self.basisND(xg, self.Ns_[ig], self.dNs_[ig])
            self.factor[ig] = self.Jacobian(self.dNs_[ig])*np.prod(wg)
            ig += 1
            
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
        
    def values_at(self, x, val = 'u'):
        """
        Return u, v, a, gradu at position x (in natural coordinates)
        """
        assert x <= 1.0 + 1.0e-14 and x >= -1.0 - 1.0e-14, 'out of element'
        N_ = np.zeros(self.Nnod,self.dtype)
        dN_ = np.zeros((self.Ndim,self.Nnod),self.dtype)
        self.basisND(x,N_,dN_)
        if val == 'u':
            u_ = np.zeros(self.Ndof,self.dtype)
            self.getU(u_)
            return u_
        if val == 'gradu':
            gradu = np.zeros(self.Ndof,self.dtype)
            self.getGradUP(gradu)
            return gradu
        if val == 'v':
            if self.timeOrder > 0:
                v_ = np.zeros(self.Ndof,self.dtype)
                self.getV(v_)
                return v_
            else:
                raise Exception('No velocity')
        if val == 'a':
            if self.timeOrder > 0:
                a_ = np.zeros(self.Ndof,self.dtype)
                self.getV(a_)
                return a_
            else:
                raise Exception('No acceleration')
        
        
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
        vGlob = data.getRe()
        vGlobD = None
        #vGlobD = data.getRiD()
        
        GaussPoints = self.intData
        R = np.zeros(self.Ndof,self.dtype)
        self.ig = -1
        t = data.getTime()
            
        # loop over Gaussian points
        for xg, wg in GaussPoints:
            self.ig += 1
            self.getBasis()
            self.material.calculate(self)
            for i in range(self.Nnod):
                try:
                    self.calculateRe(R,i,t)
                    assembleVector(vGlob, vGlobD, R, self.Nodes[i])
                except AttributeError:
                    pass
    
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
        t = data.getTime()
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
                
    def plot(self, fig = None, col = '-b', fill_mat = False, number = None):
        """
        Default plot method
        This method simply plot the nodes continuously
        Derived classes should override this method to get desired shape
        """
        if fig is None:
            fig = pl.figure()
        x = [n.getX()[0] for n in self.Nodes]
        if self.Ndim > 1:
            y = [n.getX()[1] for n in self.Nodes]
        #if self.Ndim == 3:
        #    z = [n.getX()[2] for n in self.Nodes]
        if self.Ndim == 1:
            return pl.plot(np.array(x),col),[n.getX() for n in self.Nodes]
        if self.Ndim == 2:
            return pl.plot(np.array(x),np.array(y),col),\
            [n.getX() for n in self.Nodes]
        if self.Ndim == 3:
            """
            Will be implemented later
            """
            return None
            

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
                    mGlobD[idx,-jdx-1] += mLoc[i,j]
                    
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
            vGlobD[-idx - 1] += vLoc[i]
    
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
        
    