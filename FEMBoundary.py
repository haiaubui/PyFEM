# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:52:04 2017

@author: haiau
"""

import numpy as np
import FEMElement as FE

class StandardBoundary(FE.StandardElement):
    """
    Standard Boundary element
    Derived from Standard Element
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, intData,\
    intSingData, normv, ide, intExtSingData = None,\
    dtype = 'float64', commonData = None, ndime = 1):
        """
        Initialize an Standard Boundary Element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunction: basis function, see super class info
            nodeOrder: the order of nodes in elements, array like
            intData: integration data
            intSingData: singular integration data, list of two integration
               data, one for inside element, one for outside lement
            normv: normal vector
            ide: identify of this element, use for comparison
            intExtSingData: near singular integration data
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
            ndime: 2: 2-d element
                   1: 1-d element
        Notice: Boundary Element is linear. However, a nonlinear element
        derived from this element is possible.
        """
        FE.StandardElement.__init__(self, Nodes, pd, basisFunction, nodeOrder,\
        None, intData, dtype, commonData, ndime)
        self.ide = ide
        self.nodeOrder = nodeOrder
        self.igx = 0
        self.intSingData = intSingData
        self.intExtSingData = intExtSingData
        #self.factorx = [0.0]*self.intSingData.getNumberPoint()
        self.detJ = 1.0
        self.normv = normv
        self.current = False
        self.G = 0.0
        self.gradG = np.zeros(self.Ndim,dtype)
        self.Kx = np.zeros((self.Ndof,self.Ndof),self.dtype)
        self.Rx = np.zeros(self.Ndof,self.dtype)
        self.Dx = None
        self.Mx = None
        if self.timeOrder > 0:
            self.Dx = np.zeros((self.Ndof,self.Ndof),self.dtype)
        if self.timeOrder == 2:
            self.Mx = np.zeros((self.Ndof,self.Ndof),self.dtype)
        self.xx_ = np.zeros(self.Ndim,dtype)
        self.ux_ = np.zeros(self.Ndof,dtype)
        self.vx_ = np.zeros(self.Ndof,dtype)
        self.ax_ = np.zeros(self.Ndof,dtype)
        self.gradux_ = np.zeros((self.Ndim,self.Ndof),dtype)
        self.Nx_ = None
        self.dNx_ = None
        self.Nsx_ = []
        if intSingData is None:
            ng = intData.getNumberPoint()
        else:
            ng = intSingData.getNumberPoint()
        for i in range(ng):
            self.Nsx_.append(np.zeros(self.Nnod))
        self.dNsx_ = []   
        for i in range(ng):
            self.dNsx_.append(np.zeros((self.Ndim,self.Nnod)))   
        self.prepareExtSingData()
        self.calculateBasisX(self.nodeOrder)
        #self.temp_weight = 0.0
        self.sing_int_type = 0
        self.linear = True
        
    def getSingIntData(self, element):
        if self.ide == element.ide:
            if self.intSingData is None:
                self.sing_int_type = 0
                return self.intData
            self.sing_int_type = 1
            return self.intSingData
        if self.intExtSingData is None:
            self.sing_int_type = 0
            return self.intData
        self.sing_int_type = 2
        return self.intExtSingData
        
    def prepareExtSingData(self):
        if self.intExtSingData is None:
            self.dNsex_ = None
            self.Nsex_ = None
            self.factorxx = None
            return
        Nsing = self.intExtSingData.xg.shape[0]
        Ng = self.intExtSingData.getNumberPoint()
        self.Nsex_=[]
        #self.factorxx = [[0.0]*Ng]*Nsing
        for i in range(Nsing):
            self.Nsex_.append([])
            self.dNsex_.append([])
            xg = self.intExtSingData.xg[i,:]
            #wg = self.intExtSingData.wg[i,:]
            for j in range(Ng):
                self.Nsex_[i].append(np.zeros(self.Nnod))
                self.dNsex_[i].append(np.zeros((self.Ndim,self.Nnod)))
                self.basisND(xg[j],self.Nsex_[i][j],self.dNsex_[i][j])
                self.detJ = self.Jacobian(self.dNsex_[i][j])
                #self.factorxx[i][j] = wg[i,j]            

    def getIdentifyNumber(self):
        return self.ide
        
    def setNormalVector(self, normv):
        self.normv = normv
        
    def getNormalVector(self):
        return self.normv
        
    def __eq__(self,other):
        try:
            return self.ide == other.getIdentifyNumber()
        except AttributeError:
            return False

    def calculateBasisX(self, NodeOrder):
        ig = 0
        #Nsing = self.intData.getNumberPoint()
        #ng = self.intSingData.getNumberPoint()
        #self.factorx = [[0.0]*ng]*Nsing
        #self.factorxr = [[0.0]*ng]*Nsing
        if self.intSingData is None:
            intData = self.intData
        else:
            intData = self.intSingData
        for xg, wg in intData:
            self.basisND(xg, self.Nsx_[ig], self.dNsx_[ig])
            self.detJ = self.Jacobian(self.dNsx_[ig])
            #wgx = self.intSingData.wg
            #wgxx = self.intSingData.wgx
            #for i in range(Nsing):
            #    self.factorx[i][ig] = wgx[i][ig]
            #    self.factorxr[i][ig] = wgxx[i][ig]
            ig += 1
            
    def getBasisX(self):
        if self.sing_int_type == 0:
            self.dNx_ = self.dNs_[self.igx]
            self.Nx_ = self.Ns_[self.igx]
            return
        elif self.sing_int_type == 1:
            self.dNx_ = self.dNsx_[self.igx]
            self.Nx_ = self.Nsx_[self.igx]
            return
        elif self.sing_int_type == 2:
            self.dNx_ = self.dNsex_[self.ig][self.igx]
            self.Nx_ = self.Nsex_[self.ig][self.igx]
            return
        else:
            raise Exception('Unknown Integration type')
        
    def getFactorX(self, detJ):
        if self.sing_int_type == 0:
            return self.factor[self.igx]
        elif self.sing_int_type == 1:
            try:
                return self.intSingData.wg[self.ig][self.igx]*detJ
            except:
                return self.intSingData.wg[self.igx]*detJ
        elif self.sing_int_type == 2:
            return self.intExtSingData.wg[self.ig,self.igx]*detJ
        else:
            raise Exception('Unknown Integration type')
            
    def getFactorXR(self, detJ):
        if self.sing_int_type == 0:
            return self.factor[self.igx]
        elif self.sing_int_type == 1:
            try:
                return self.intSingData.wgx[self.ig][self.igx]*detJ
            except:
                return self.intSingData.wg[self.igx]*detJ
        elif self.sing_int_type == 2:
            return self.intExtSingData.wg[self.ig,self.igx]*detJ
        else:
            raise Exception('Unknown Integration type')
        
    def getAllValuesX(self, data, current = False, mechDofs = None):
        """
        Get all the current values: displacement, velocity and acceleration
        Input:
            data: a class that have methods return global vectors
            current: current configuration
            mechDofs: an array contains all mechanical dof indices
        """
        # Calculate coordinate at current Gaussian point
        self.xx_.fill(0.0)
        self.getX(self.xx_, self.Nx_)
        
        # calculate displacement
        self.ux_.fill(0.0)            
        self.getU(self.ux_, self.Nx_)
        
        if current:
            if not mechDofs is None:
                for i in range(len(mechDofs)):
                    self.xx_[i] += self.u_[mechDofs[i]]
        
        # calculate gradient of displacement
        self.gradux_.fill(0.0)
        self.getGradU(self.gradux_,data,self.dNx_)
        
        # calculate velocity
        if data.getTimeOrder() > 0:
            self.vx_.fill(0.0)
            self.getV(self.vx_,self.Nx_)
            
        # calculate acceleration
        if data.getTimeOrder() == 2:
            self.ax_.fill(0.0)
            self.getA(self.ax_,self.Nx_)
            
    def calculateGreen(self, r, rp, gradG):
        pass
    
    def subCalculate(self, data, element, Nodei, i, linear):
        """
        Calculate second integration
        """
        # Get matrices from data
        timeOrder = data.getTimeOrder()
        #vGlob,vGlobD = data.getRi(),data.getRid()
        kGlob,kGlobD = data.getKtL(),data.getKtLd()
        dGlob,dGlobD = data.getDL(),data.getDLd()
        mGlob,mGlobD = data.getML(),data.getMLd()
        
        #Gauss integration points
        GaussPoints = element.getSingIntData(self)
        
        # initialized matrices and vector
        R = self.R
        K = self.Kx
        K.fill(0.0)
        D = self.Dx
        M = self.Mx
            
        for element.igx in range(GaussPoints.getNumberPoint()):
            #self.temp_weight = wg
            
            #self.calculateBasis(xg, self.nodeOrder)
            #self.factor = self.Jacobian(self.dN_)
            element.getBasisX()
            
            # Calculate coordinate at current Gaussian point
            element.getAllValuesX(data,element.current)
            
            try:
                self.calculateGreen(self.x_,element.xx_)
                #self.G = 1.0
                #self.gradG[0] = 0.0
                #self.gradG[1] = 0.0
            except SingularPoint:
                continue
            # Initialize matrices
            #self.initializeMatrices()
            
            #self.material.calculate(self)
            
            # loop over node j
            if linear:
                try:
                    for j in range(self.Nnod):
                        # calculate and assemble matrices
                        self.subCalculateKLinear(K, element, i, j)
                        FE.assembleMatrix(kGlob,kGlobD,K,\
                        Nodei,element.Nodes[j])
                except:
                    pass
                if timeOrder > 0:
                    try:
                        self.subCalculateDLinear(D, element, i, j)
                        FE.assembleMatrix(dGlob,dGlobD,D,Nodei,\
                        element.Nodes[j])
                    except:
                        pass
                if timeOrder == 2:
                    try:
                        self.subCalculateMLinear(M, element, i, j)
                        FE.assembleMatrix(mGlob,mGlobD,M,Nodei,\
                        element.Nodes[j])
                    except:
                        pass
                continue
            
            try:
                self.subCalculateR(R[i],element,i)
            except:
                pass
            #FE.assembleVector(vGlob, vGlobD, R, Nodei)
            try:
                for j in range(self.Nnod):
                    self.subCalculateK(K, element, i, j)
                    FE.assembleMatrix(kGlob,kGlobD,K,\
                        Nodei,element.Nodes[j])
            except:
                pass
            if timeOrder > 0:
                try:
                    for j in range(self.Nnod):
                        self.subCalculateD(D, element, i, j)
                        FE.assembleMatrix(dGlob,dGlobD,D,Nodei,\
                        element.Nodes[j])
                except:
                    pass
            if timeOrder == 2:
                try:
                    for j in range(self.Nnod):
                        self.subCalculateM(M, element, i, j)
                        FE.assembleMatrix(mGlob,mGlobD,M,Nodei,\
                        element.Nodes[j])
                except:
                    pass
        
    def calculate(self, data, linear = False):
        """
        Calculate matrices and vectors
        Input:
            data: a class that have methods return global matrices and vectors
            linear: linearity, True: only linear part will be calculated
                               False: only nonlinear part will be calculated
        """
        # Get matrices from data
        t = data.getTime()
        timeOrder = data.getTimeOrder()
        #vGlob,vGlobD = data.getRi(),data.getRid()
        if linear:
            kGlob,kGlobD = data.getKtL(),data.getKtLd()
            dGlob,dGlobD = data.getDL(),data.getDLd()
            mGlob,mGlobD = data.getML(),data.getMLd()
        
        otherB = data.getMesh().getBoundaryElements()
        
        #make a copy of this object to calculate second integration
        
        #thisE = deepcopy(self)
        
        #Gauss integration points
        GaussPoints = self.intData
        #SingData = self.intSingData
        
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
            
        for self.ig in range(GaussPoints.getNumberPoint()):
            self.getBasis()
            
            # Calculate coordinate at current Gaussian point
            self.getAllValues(data,self.current)
            
            # Initialize matrices
            #self.initializeMatrices()
            
            try:
                self.material.calculate(self)
            except AttributeError:
                pass
            if linear:
                for i in range(self.Nnod):
                    # calculate and assemble load vector
                    #self.calculateR(R,i,t)
                    #assembleVector(vGlob, vGlobD, R, self.Nodes[i])
                    
                    # loop over node j
                    try:
                        for j in range(self.Nnod):
                            # calculate and assemble matrices
                            self.calculateKLinear(K,i,j,t)
                            FE.assembleMatrix(kGlob,kGlobD,K,\
                            self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                    try:
                        if timeOrder > 0:
                            for j in range(self.Nnod):
                                self.calculateDLinear(D,i,j,t)
                                FE.assembleMatrix(dGlob,dGlobD,D,\
                                self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                    try:                        
                        if timeOrder == 2:
                            for j in range(self.Nnod):
                                self.calculateMLinear(M,i,j,t)
                                FE.assembleMatrix(mGlob,mGlobD,M,\
                                self.Nodes[i],self.Nodes[j])
                    except AttributeError:
                        pass
                    # loop over orther boundary elements
                    for belement in otherB:
                        self.subCalculate(data, belement, self.Nodes[i],\
                        i, linear)
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
                    self.calculateK(K[i][j],i,0,t)
                    for j in range(1,self.Nnod):
                        # calculate and assemble matrices
                        self.calculateK(K[i][j],i,j,t)
                except AttributeError:
                    pass
                try:
                    self.calculateD(D[i][j],i,0,t)
                    for j in range(1,self.Nnod):
                        self.calculateD(D[i][j],i,j,t)
                except (AttributeError, TypeError):
                    pass
                try:
                    self.calculateM(M[i][j],i,0,t)                      
                    for j in range(1,self.Nnod):
                        self.calculateM(M[i][j],i,j,t)
                except (AttributeError, TypeError):
                    pass
                        
                # loop over orther boundary elements
                for belement in otherB:
                    if belement == self:
                        continue
                    self.subCalculate(data, belement, self.Nodes[i],\
                    i, linear)
                self.subCalculate(data, self, self.Nodes[i],\
                i, linear)    
                
    def postCalculate(self, x_p, intDat = None):
        """
        Calculate value at some point x_p after Finite Element Analysis
        The mesh has to be updated with values before calling this method
        """
        if intDat is None:
            intDatx = self.intData
        else:
            intDatx = intDat
            
        N_ = np.zeros(self.Nnod)
        dN_ = np.zeros((self.Ndim,self.Nnod))
        
        res = np.zeros(self.Ndof)
            
        for xg,wg in intDatx:
            self.basisND(xg, N_, dN_)
            factor = self.Jacobian(dN_)*np.prod(wg)
            
            self.x_.fill(0.0)
            self.getX(self.x_,N_)
            self.gradu_.fill(0.0)
            self.getGradUP(self.gradu_,dN_)
            self.u_.fill(0.0)
            self.getU(self.u_,N_)
            self.getV(self.v_,N_)
            self.getA(self.a_,N_)
            
            try:
                self.calculateGreen(x_p,self.x_)
            except SingularPoint:
                continue
            
            self.postCalculateF(N_,dN_,factor,res)
            
        return res
        

# End of StandardBoundary class definition

class StandardStaticBoundary(StandardBoundary):
    """
    Standard static boundary is a boundary element that is not moving during
    simulation.
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, intData,\
    intSingData, normv, ide, intExtSingData = None,\
    dtype = 'float64', commonData = None):
        """
        Initialize an Standard Boundary Element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunction: basis function, see super class info
            nodeOrder: the order of nodes in elements, array like
            intData: integration data
            intSingData: singular integration data, list of two integration
               data, one for inside element, one for outside lement
            normv: normal vector
            ide: identify of this element, use for comparison
            intExtSingData: near singular integration data
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
        """
        StandardBoundary.__init__(self,Nodes,pd,basisFunction,nodeOrder,\
        intData,intSingData,normv,ide,intExtSingData,dtype,commonData)
        self.current = False
        
class StandardMovingBoundary(StandardBoundary):
    """
    Standard static boundary is a boundary element that is not moving during
    simulation.
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, intData,\
    intSingData, normv, ide, intExtSingData = None,\
    dtype = 'float64', commonData = None):
        """
        Initialize an Standard Boundary Element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
                if number of dimension == 1, pd is an integer
                else, pd is an list
            basisFunction: basis function, see super class info
            nodeOrder: the order of nodes in elements, array like
            intData: integration data
            intSingData: singular integration data, list of two integration
               data, one for inside element, one for outside lement
            normv: normal vector
            ide: identify of this element, use for comparison
            dtype: datatype of arrays, default = 'float64'
            commonData: common data shared between elements
        """
        StandardBoundary.__init__(self,Nodes,pd,basisFunction,nodeOrder,\
        intData,intSingData,normv,ide,intExtSingData,dtype,commonData)
        self.current = True
        self.dgradG = np.zeros((self.Ndim,self.Ndim),self.dtype)

class SingularPoint(Exception):
    """
    Exception for singular point integration
    """
    pass
    