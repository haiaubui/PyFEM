# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:19:01 2017

@author: haiau
"""
import itertools as it
import FEMElement as FE
import numpy as np
import pylab as pl
import math

class QuadElement(FE.StandardElement):
    """
    Quadrilateral Element
    Derived from standard element
    """
    def basisND(self, x_, N_, dN_):
        """
        n-dimensional basis functions, n = 1, n = 2 or n = 3
        n = self.Ndim
        """            
        if self.nodeOrder is None:
            self.nodeOrder = generateQuadNodeOrder(self.pd, self.Ndim)
        FE.StandardElement.basisND(self, x_, N_, dN_)


class LagrangeElement1D(QuadElement):
    """
    1-dimensional Lagrange Elment
    """
    def __init__(self, Nodes, material, intData):
        """
        Initialize 1-dimensional Lagrange element
        Input:
            Nodes: nodes of elements
            intData: integration data
        """
        QuadElement.__init__(self,Nodes,len(Nodes)-1,\
        LagrangeBasis1D,None,material,intData)
        
        
class LagrangeElement2D(QuadElement):
    """
    2-dimensional Lagrange Elment
    """
    def __init__(self, Nodes, pd, nodeOrder, material, intData):
        """
        Initialize an Lagrange Element
        Input:
            Nodes: nodes of elements
            pd: basis function degree(s),
            nodeOrder: the order of nodes in elements,
            material: material of element
            intData: integration data
        """
        QuadElement.__init__(self,Nodes,pd,\
        LagrangeBasis1D,nodeOrder,material,intData)
        
class Quad9Element(QuadElement):
    """
    Quad9 element: quadrilateral element with 9 nodes
    """
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
        
class Quad9Flat(Quad9Element):
    """
    Flat Quad9 element: quadrilateral element with 9 nodes and all edges are
    straight lines.
    """
#    def getXi(self, x, N_ = None, dN_ = None, xi = None, max_iter = 100,\
#    rtol = 1.0e-8):
#        """
#        Return natural coordinate xi corresponding to physical coordinate x
#        N_: array to store shape functions
#        dN_: array to store derivatives of shape functions
#        max_iter: maximum number of iterations for Newton method
#        Raise OutsideEdlement if x is not inside element.
#        """
#        if x[0] > self.maxx and x[0] < self.minx\
#        and x[1] > self.maxy and x[1] < self.miny:
#            raise FE.OutsideElement
#        
#        for i in range(9):
#            n1 = self.nodeOrder[0][i]
#            n2 = self.nodeOrder[1][i]
#            if n1 == 0 and n2 == 0:
#                X1 = self.Nodes[i].getX()
#            if n1 == 0 and n2 == 2:
#                X3 = self.Nodes[i].getX()
#            if n1 == 2 and n2 == 0:
#                X2 = self.Nodes[i].getX()
#            if n1 == 2 and n2 == 2:
#                X4 = self.Nodes[i].getX()
#        
#        A = X1[0] + X2[0] + X3[0] + X4[0]
#        B = X1[0] - X2[0] - X3[0] + X4[0]
#        C = -X1[0] + X2[0] - X3[0] + X4[0]
#        D = -X1[0] - X2[0] + X3[0] + X4[0]
#        E = X1[1] + X2[1] + X3[1] + X4[1]
#        F = X1[1] - X2[1] - X3[1] + X4[1]
#        G = -X1[1] + X2[1] - X3[1] + X4[1]
#        H = -X1[1] - X2[1] + X3[1] + X4[1]
#        
#        if xi is None:
#            xi = np.zeros(2,self.dtype)
#        
#        alpha = B*H + F*D
#        beta = B*E + F*x[0] - F*A - G*D + H*C
#        gamma = E*C + G*x[0] - G*A - x[1]
#        if math.fabs(alpha) < 1.0e-14:
#            if math.fabs(beta) < 1.0e-14:
#                raise FE.ElementError
#            xi[1] = -gamma/beta
#        else:
#            delta = beta*beta - 4.0*alpha*gamma
#            if delta < -1.0e-15:
#                raise FE.ElementError
#            
#            xi1 = (-beta + math.sqrt(delta))/(2.0*alpha)
#            xi2 = (-beta - math.sqrt(delta))/(2.0*alpha)
#            if math.fabs(xi1) > 1.0+1.0e-14:
#                xi[1] = xi1
#            elif math.fabs(xi2) > 1.0+1.0e-14:
#                xi[1] = xi2
#            else:
#                raise FE.OutsideElement
#        denom = x[1]*B + C
#        if np.isinf(denom):
#            raise FE.ElementError
#        if math.fabs(denom) < 1.0e-15:
#            raise FE.OutsideElement
#        xi[0] = (x[0] - A - D*xi[1])/denom
#        if math.fabs(xi[0]) > 1.0+1.0e-14:
#            raise FE.OutsideElement
#            
#        return xi
        
                
        
def LagrangeBasis1D(x_, pd, N_ = None, dN_ = None, Order = None):
    """
    Calculate Lagrange Basis functions and its derivatives
    Input:
        x_ : parametric coordinates
        pd : polynomial degree
        N_= None : create new array for N_
        dN_ = None: create new array for dN_
    Return: the updated arrays N_ and dN_
    """
    n = pd + 1
    if N_ is None:
        N_ = np.empty(n)
    if dN_ is None:
        dN_ = np.empty(n)
        
    if pd == 1:
        N_[1] = 0.5*(x_ + 1.0)
        N_[0] = 0.5*(1.0 - x_)
        dN_[0] = -0.5
        dN_[1] = 0.5
    elif pd == 2:
        N_[0] = 0.5*x_*(x_ - 1.0)
        N_[1] = 1.0 - x_*x_
        N_[2] = 0.5*x_*(1.0 + x_)
        dN_[0] = x_ - 0.5
        dN_[1] = -2.0*x_
        dN_[2] = 0.5 + x_
    else:
        pass
    
    return N_, dN_
        
def generateQuadNodeOrder(pd, ndim, typeE = 'rcd'):
    """
    Generate Node Order for Quadrilateral Element
    Input:
        pd: polynomial degree(s), array or scalar
        ndim: number of dimension, 1 <= ndim <= 3
        typeE = 'rc',  default node in row-column order
                'cr',                  column-row
                'rcd',                 row-column-depth
                and so on
    Return node order list.
    """
    if isinstance(pd, str):
        raise Exception('first parameters cannot be a string')
    if ndim == 1:
        try:
            p = pd[0]
        except (TypeError, IndexError):
            p = pd
        return list(range(p+1))
    if ndim == 2:
        n1,n2 = pd[0] + 1,pd[1] + 1
        order = \
        np.array(list(it.product(range(n1),range(n2)))).transpose().tolist()
        if typeE[0:2] == 'rc':
            return list(reversed(order))
        elif typeE[0:2] == 'cr':
            return order
        raise Exception('Cannot generate node order of type '+str(typeE))
        
    if ndim == 3:
        n1,n2,n3 = pd[0] + 1,pd[1] + 1,pd[2] + 1
        order = np.array(list(it.product(range(n1),range(n2),range(n3))))
        order = order.transpose().tolist()
        if typeE == 'rcd':
            return order
        elif typeE == 'crd':
            return [order[1],order[0],order[2]]
        elif typeE == 'rdc':
            return [order[0],order[2],order[1]]
        elif typeE == 'cdr':
            return [order[1],order[2],order[0]]
        elif typeE == 'dcr':
            return [order[2],order[1],order[0]]
        elif typeE == 'drc':
            return [order[2],order[0],order[1]]
        raise Exception('Cannot generate node order of type '+str(typeE))
            

    
                        
                    