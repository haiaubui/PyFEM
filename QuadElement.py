# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:19:01 2017

@author: haiau
"""
import itertools as it
import FEMElement as FE
import numpy as np

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

class QuadBoundary(FE.StandardBoundary):
    """
    Boundary of Quadrilateral element
    """
    def calculateBasis(self, x_, NodeOrder):
        return QuadElement.basisND(self,x_)


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
        N_[1] = (1.0 - x_)*(1.0 + x_)
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
        if isinstance(pd, (list,np.array,tuple)):
            p = pd[0]
        else:
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
            

    
                        
                    