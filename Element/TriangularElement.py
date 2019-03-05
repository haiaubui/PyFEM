# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:22:19 2018

@author: haiau
"""

import Element.FEMElement as FE
import Math.IntegrationData as IntDat
import numpy as np

class T6Element(FE.StandardElement):
    """
    T6 element (Triangular element with 6 nodes)
    The standard nodes order is following
                   *1
                  / \
                 /   \
               5*     *4
               /       \
              /         \
            2*-----*6----*3
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
        FE.StandardElement.__init__(self,Nodes,[2,2],None,None,material,\
        TriangularGaussian(nG),dtype,commonData,2)
    
    def basisND(self, x_, N_, dN_):
        """
        NOTICE: x_ will be area coordinates, i.e. size(x_) = 3
        n-dimensional basis functions, n = 1, n = 2 or n = 3
        n = self.Ndim
        """ 
        sz = 6
        if N_ is None:            
            N_ = np.empty(sz,self.dtype)
        if dN_ is None:
            dN_ = np.empty((self.Ndim,sz),self.dtype)
            
        N_[0] = 0.5*x_[0]*(x_[0]+1.0)
        dN_[0,0] = -(x_[0]+0.5)
        dN_[1,0] = dN_[0,0]
        N_[1] = 0.5*x_[1]*(x_[1]+1.0)
        dN_[0,1] = x_[1]+0.5
        dN_[1,1] = 0.0
        N_[2] = 0.5*x_[2]*(x_[2]+1.0)
        dN_[1,2] = x_[2]+0.5
        dN_[0,2] = 0.0
        N21eta = 0.5*(1.0+x_[0])
        dN21eta = 0.5
        N21xi1 = 0.5*(1.0+x_[1])
        dN21xi1 = 0.5
        N21xi2 = 0.5*(1.0+x_[2])
        dN21xi2 = 0.5
        N_[3] = 4.0*N21eta*N21xi2
        dN_[0,3] = -4.0*dN21eta*N21xi2
        dN_[1,3] = -4.0*dN21eta*N21xi2 + 4.0*N21eta*dN21xi2
        N_[4] = 4.0*N21eta*N21xi1
        dN_[0,4] = -4.0*dN21eta*N21xi1 + 4.0*N21eta*dN21xi1
        dN_[1,4] = -4.0*dN21eta*N21xi1
        N_[5] = 4.0*N21xi1*N21xi2
        dN_[0,5] = 4.0*dN21xi1*N21xi2
        dN_[1,5] = 4.0*N21xi1*dN21xi2            
    
        return N_, dN_

class TriangularGaussian(IntDat.GaussianQuadrature):
    """
    Integration Data for Triangular element
    """
    def __init__(self, npoint):
        assert npoint==1 or npoint==3 or npoint==4 or npoint==7,\
        'unsupport number of points'
        IntDat.GaussianQuadrature.__init__(self,npoint,3,None)
        self.xg = np.zeros((3,npoint),dtype='float64')
        self.wg = np.zeros(npoint,dtype='float64')
        if npoint == 1:
            self.xg[:,0] = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            self.wg[0] = 1.0
        elif  npoint == 3:
            self.xg[:,0] = [0.5, 0.5, 0.0]
            self.wg[0] = 1.0/3.0
            self.xg[:,1] = [0.5, 0.0, 0.5]
            self.wg[1] = 1.0/3.0
            self.xg[:,2] = [0.0, 0.5, 0.5]
            self.wg[2] = 1.0/3.0
        elif npoint == 4:
            self.xg[:,0] = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            self.wg[0] = -27.0/48.0
            self.xg[:,1] = [0.6, 0.2, 0.2]
            self.wg[1] = 25.0/48.0
            self.xg[:,2] = [0.2, 0.6, 0.2]
            self.wg[2] = 25.0/48.0
            self.xg[:,3] = [0.2, 0.2, 0.6]
            self.wg[3] = 25.0/48.0
        elif npoint == 7:
            a1 = 0.0597158717
            b1 = 0.4701420641
            a2 = 0.7974269853
            b2 = 0.1012865073
            self.xg[:,0] = [1.0/3.0, 1.0/3.0, 1.0/3.0]
            self.wg[0] = 0.225
            self.xg[:,1] = [a1, b1, b1]
            self.wg[1] = 0.1323941527
            self.xg[:,2] = [b1, a1, b1]
            self.wg[2] = 0.1323941527
            self.xg[:,3] = [b1, b1, a1]
            self.wg[3] = 0.1323941527
            self.xg[:,4] = [a2, b2, b2]
            self.wg[4] = 0.1259391805
            self.xg[:,5] = [b2, a2, b2]
            self.wg[5] = 0.1259391805
            self.xg[:,6] = [b2, b2, a2]
            self.wg[6] = 0.1259391805
            
        self.xg *= 2.0
        self.xg -= 1.0
        