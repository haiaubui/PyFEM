# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:54:02 2017

@author: haiau
"""

import math
import numpy as np
import scipy.special as scs
import IntegrationData as Idat

__all__ = ['SingularGaussian1D','Gaussian_1D_log','Gaussian_1D_rat',
           'Gaussian_1D_Pn_Log','Gaussian_1D_Pn_Log_Rat']

class SingularGaussian1D(Idat.GaussianQuadrature):
    """
    Gaussian Quadrature data for Singular function in one dimension
    """
    def __init__(self, ng, sing_points, gen1, gen2 = None):
        Idat.GaussianQuadrature.__init__(self,ng,1,None)
        self.Nsing = len(sing_points)
        self.wg = []
        self.wgx = None
        self.gen1 = gen1
        self.gen2 = gen2
        if gen2 is not None:
            self.wgx = []
            
        for t in sing_points:
            self.xg,wg = gen1(t, ng)
            if gen2 is not None:
                _,wgx = gen2(t,ng)
            self.wg.append(wg)
            self.wgx.append(wgx)
            
        if gen2 is None:
            self.wgx = self.wg
            
    def __next__(self):
        if self.iter_index == self.Npoint:
            raise StopIteration
        idx = self.iter_index
        self.iter_index += 1
        if self.Ndim == 1:
            return self.xg[idx], self.wg[0][idx]
        return self.xg[:,idx], self.wg[0][:,idx]
            
    def s_iter(self, op):
        """
        second iterator
        Notice: this generator yields one Gaussian point and two weights at one
        loop step.
        """
        for i in range(self.Ng):
            yield self.xg[i], self.wg[op][i], self.wgx[op][i]

def Gaussian_1D_log(t, ng):
    """
    Gaussian quadrature points and weights for logarithmic function log|x-t|
    integration
    """
    x,w = Idat.Gaussian1D(ng)
    w1 = np.zeros(ng)
    for i in range(ng):
        for j in range(1,ng-1):
            w1[i] += (poly_Legendre(x[i],j-1)-poly_Legendre(x[i],j+1))*\
            R_function(t,j)
            
        w1[i] += (poly_Legendre(x[i],0)-poly_Legendre(x[i],1))*R_function(t,0)
        w1[i] += poly_Legendre(x[i],ng-2)*R_function(t,ng-1)
        w1[i] += poly_Legendre(x[i],ng-1)*R_function(t,ng)
        w1[i] *= w[i]
    return x, w1
    
def Gaussian_1D_rat(t, ng):
    """
    Gaussian quadrature points and weights for rational function 1/|x-t|
    integration
    """
    x,w = Idat.Gaussian1D(ng)
    w1 = np.zeros(ng)
    for i in range(ng):
        for j in range(ng-1):
            w1[i] += (2.0*j+1.0)*poly_Legendre(x[i],j)*Legendre_Qn(t,j)
        w1[i] *= w[i]
        
    return x, w1
    
def integration_PnLog(t, pn):
    x,w = Gaussian_1D_log(t, 20)
    y = 0.0
    for i in range(20):
        y += w[i]*poly_Legendre(x[i],pn)
    return y
    
def Gaussian_1D_Pn_Log(t, ng, m = -1):
    """
    Gaussian quadrature points and weights for function Pn(x)*log|x-t|
    """
    if m <= 0:
        m = math.ceil(ng/3)
    x,_ = Idat.Gaussian1D(ng)
    xi = np.empty((2*m,ng))
    mi = np.zeros(2*m)
    mi[0] = 2.0
    
    for i in range(m):
        for j in range(ng):
            pol = poly_Legendre(x[j],i)
            xi[i,j] = pol
            xi[i+m,j] = pol*math.log(math.fabs(x[j]-t))
        mi[i+m] = integration_PnLog(t,i)
        
    w1,_,_,_ = np.linalg.lstsq(xi, mi)
    
    return x, w1
    
def integration_PnRat(t, pn):
    x,w = Gaussian_1D_rat(t, 20)
    y = 0.0
    for i in range(20):
        y += w[i]*poly_Legendre(x[i],pn)
        
    return y
    
def Gaussian_1D_Pn_Log_Rat(t, ng, m = -1):
    if m <= 0:
        m = math.ceil(ng/3)
    x,_ = Idat.Gaussian1D(ng)
    xi = np.empty((3*m,ng))
    mi = np.zeros(3*m)
    mi[0] = 2.0
    
    for i in range(m):
        for j in range(ng):
            pol = poly_Legendre(x[j],i)
            xi[i,j] = pol
            xi[i+m,j] = pol*math.log(math.fabs(x[j]-t))
            xi[i+2*m,j] = pol/(t-x[j])
        mi[i+m] = integration_PnLog(t,i)
        mi[i+2*m] = integration_PnRat(t,i)
        
    w1,_,_,_ = np.linalg.lstsq(xi, mi)
    return x, w1


def poly_Legendre(x, n):
    p = scs.legendre(n)
    return p(x)
    
def Legendre_Qn(x, n):
    yn_1 = 0.5*math.log((1.0+x)/(1.0-x))
    yn = x*0.5*math.log((1.0+x)/(1.0-x))-1.0
    if n == 0:
        y = yn_1
    elif n == 1:
        y = yn
    else:
        for i in range(1,n):
            y = ((2.0*i+1.0)*x*yn-i*yn_1)/(i+1.0)
            yn_1 = yn
            yn = y
    return y
    
def R_function(x, n):
    return Legendre_Qn(x,n) + 0.25*math.log((x-1.0)*(x-1.0))
    
if __name__ == '__main__':
    print(poly_Legendre(0.5,3))
    print(R_function(0.5,3))