# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:10:50 2017

@author: haiau
"""

import numpy as np
import scipy.linalg as la

class Solver(object):
    """
    Matrix solver
    Different solver can be used by deriving a subclass of this class
    and overwrite solve method
    """
    def solve(self, A, b):
        """
        solve system of equations Ax = b
        return x as solution
        Raise SingularMatrix if matrix is singular
        This method is often used in linear algorithms
        """
        pass
    
    def isolve(self, A, b):
        """
        Solve system of equation Ax = b
        replace b by solution x after calculation
        Raise SingularMatrix if matrix is singular
        This method is often used in nonlinear algorithms
        """
        pass

class numpySolver(Solver):
    """
    numpy and scipy solver
    """    
    def solve(self, A, b):
        """
        solve system of equations Ax = b
        return x as solution
        Raise SingularMatrix if matrix is singular
        This method is often used in linear algorithms
        """
        try:
            return np.linalg.solve(A,b)
        except np.linalg.LinAlgError:
            raise SingularMatrix
            
    def isolve(self, A, b):
        """
        Solve system of equation Ax = b
        replace b by solution x after calculation
        Raise SingularMatrix if matrix is singular
        This method is often used in nonlinear algorithms
        """
        try:
            a = np.linalg.solve(A,b)
            np.copyto(b,a)
#            la.solve(A,b,overwrite_a = True, overwrite_b = True)
        except la.LinAlgError:
            raise SingularMatrix
            
    def LDLdiag(self, A):
        """
        return D of LDLdecomposition
        """
        return np.diag(la.cholesky(A))
    
    def LUdecomp(self, A):
        return la.lu(A)
            
class SingularMatrix(Exception):
    """
    Exception for Singular matrix
    """
    pass
    