# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:29:13 2017

@author: haiau
"""

import numpy as np
import QuadElement as qa

class AxisymmetricQuadElement(qa.QuadElement):
    """
    Axisymmetric Quadrilateral Element
    """
    def __init__(self, Nodes, pd, basisFunction, nodeOrder, material, intData):
        """
        Initialize an Axisymmetric Element
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
        Raise:
            DimensionMismatch: if dimensions in Nodes is not equal 2
        """
        for node in Nodes:
            if node.getNdim() != 2:
                raise DimensionMismatch
                
        qa.QuadElement.__init__(self,Nodes,pd,basisFunction,\
        nodeOrder,material,intData)
        
    def getFactor(self):
        """
        Return: factor for integration, 2*pi*radius*det(J)
        """
        return 2*np.pi*self.x_[0]*self.factor[self.ig]
        
class DimensionMismatch(Exception):
    """
    Exception for dimensions in node and element do not match
    """
    pass