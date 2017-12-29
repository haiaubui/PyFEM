# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 10:26:43 2017

@author: Hai Au Bui
         University of Kassel
@email: au.bui(_at_)uni-kassel.de
"""
import math
import numpy as np
from MeshGenerator import GeneralNode
import injectionArray as ia

class Node(GeneralNode):
    """
    Node, basic component of Finite Element Method
    This class has all of the properties and methods that are necessary for
    a FEM Node
    """
    def __init__(self, X_, Ndof, timeOrder = 0, id_number = 0,\
    dtype = 'float64'):
        """
        Initialize Node with the coordinate X and number of Degree of Fredoms
        Input: 
            X_: an array like object that store coordinates
            Ndof: number of degree of freedom
            timeOrder: optional, the order of differential quation in time
                       default value: "Zeroth"
                       other possible values: "First" and "Second"
        """
        GeneralNode.__init__(self, X_, len(X_))
        self.dtype = dtype
        self.id_number = id_number
        self.Ndof = Ndof
        # Initialize a list that stores the indices of equations
        self.ID = [-1]*Ndof
        # Initialize a list that stores the contraints of this node
        self.freedom = [True]*Ndof
        # Initialize an array that store point external load or source
        self.load = [0.0]*Ndof
        # Initialize a list that stores the diplacements, velocities
        # accelerations
        self.u_ = ia.zeros(Ndof)
        self.v_ = None
        self.a_ = None
        self.timeOrder = timeOrder
        self.callPointLoad = False
        if timeOrder > 0:
            self.v_ = ia.zeros(Ndof)
            if timeOrder > 1:
                self.a_ = ia.zeros(Ndof)
                
        self.NonHomogeneousDirichlet = False
        
    def copyToPosition(self, X):
        """
        Copy this node to some other position X
        Return node of the same type
        id_number will be 0
        This method does not copy the boundary conditions, equation numbers
        and load
        """
        n = Node(X, self.Ndof, timeOrder = self.timeOrder)
            
        return n
        
    def __str__(self):
        """
        Print out information of node
        """
        s = 'Position: ' + str(self.X_.tolist())
        s += '\n'
        s += 'Number of degree of freedoms: ' + str(self.Ndof)
        s += '\n'
        s += 'Time order: '+str(self.timeOrder)
        s += '\n'
        s += 'Equation number and Values: \n'
        s += ' \t number \t U \t V \t A: \n'
        for i in range(self.Ndof):
            s += 'DOF '+ str(i) + ':\t'+ \
            str(self.ID[i])+'\t'+"{:.3e}".format(self.u_[i])
            if self.timeOrder > 0:
                s+='\t'+"{:.3e}".format(self.v_[i])
            else:
                s+='\tNaN'
            if self.timeOrder == 2:
                s+='\t'+"{:.3e}".format(self.a_[i])
            else:
                s+='\tNaN'
            s += '\n'
        s += 'Load: \n'
        for i in range(self.Ndof):
            s += 'DOF '+ str(i) + ':\t'+ "{:.3e}".format(self.load[i])
            s += '\n'
        return s
        
    def get_id_number(self):
        return self.id_number
        
    def get_dtype(self):
        """
        Return datatype of arrays in this node
        """
        return self.dtype
        
    def getNdof(self):
        """
        Return number of degree of freedoms
        """
        return self.Ndof
        
    def getID(self):
        """
        Return the indices of equations
        """
        return self.ID
    
    def getU(self):
        """
        Return the displacements at this node
        """
        return self.u_
        
    def getV(self):
        """
        Return the velocity at this node
        """
        return self.v_
    
    def getA(self):
        """
        Return the acceleration at this node
        """
        return self.a_
        
    def getValue(self, val = 'u'):
        """
        Return value at node
        val = 'u' return displacement, variable
        val = 'v'        velocity    , time derivative of variable
        val = 'a'        acceleration, second time derivative of variable
        """
        if val == 'u':
            return self.u_
        if val == 'v':
            return self.v_
        if val == 'a':
            return self.a_
        
    def setU(self, u):
        """
        set displacement
        """
        self.u_ = ia.array(u,self.dtype)
        
    def setV(self, v):
        """
        set velocity
        """
        self.v_ = ia.array(v,self.dtype)
        
    def setA(self, a):
        """
        set acceleration
        """
        self.a_ = ia.array(a,self.dtype)
        
    def connect(self, U, V, A):
        """
        Connect global vector U,V,A to local u_,v_,a_
        """
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                self.u_.connect(i, U, idx)
                
        try:
            for i in range(self.Ndof):
                idx = self.ID[i]
                if idx >= 0:
                    self.v_.connect(i, V, idx)
        except (AttributeError, TypeError):
            pass
        
        try:
            for i in range(self.Ndof):
                idx = self.ID[i]
                if idx >= 0:
                    self.a_.connect(i, A, idx)
        except (AttributeError, TypeError):
            pass
        
    def getFromGlobalU(self, uGlob, x):
        """
        Get nodal value from global vector
        Input:
            uGlob : global vector
            x: to store result
        Return nodal value, None if uGlob == None
        """
        if not uGlob is None:
            if x is None:
                x = np.empty(self.Ndof,self.dtype)
            for i in range(self.Ndof):
                idx = self.ID[i]
                if idx >= 0:
                    x[i] = uGlob[idx]
                else:
                    x[i] = self.u_[i]
        else:
            return None
        return x
        
    def getFromGlobalV(self, uGlob, x):
        """
        Get nodal value from global vector
        Input:
            uGlob : global vector
            x: to store result
        Return nodal value, None if uGlob == None
        """
        if not uGlob is None:
            if x is None:
                x = np.empty(self.Ndof,self.dtype)
            for i in range(self.Ndof):
                idx = self.ID[i]
                if idx >= 0:
                    x[i] = uGlob[idx]
                else:
                    x[i] = self.v_[i]
        else:
            x = None
        return x
        
    def getFromGlobalA(self, uGlob, x):
        """
        Get nodal value from global vector
        Input:
            uGlob : global vector
            x: to store result
        Return nodal value, None if uGlob == None
        """
        if not uGlob is None:
            if x is None:
                x = np.empty(self.Ndof,self.dtype)
            for i in range(self.Ndof):
                idx = self.ID[i]
                if idx >= 0:
                    x[i] = uGlob[idx]
                else:
                    x[i] = self.a_[i]
        else:
            x = None
        return x    
    
    def getFromGlobal(self, data, u, v, a):
        """
        Get nodal values from global vectors
        Input:
            data: a class that have methods return global vectors
        Return nodal displacement, velocity and acceleration
               None if global displacement, velocity or acceleration is None
               correspondingly
        """
        uGlob = data.getU()
        vGlob = data.getV()
        aGlob = data.getA()
        return self.getFromGlobalU(uGlob,u), self.getFromGlobalV(vGlob,v),\
        self.getFromGlobalA(aGlob,a)
            
        
    def updateU(self, uGlobal):
        """
        update current value of displacement
        Input:
            uGlobal: global vector of displacement
        """
        if uGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                self.u_[i] = uGlobal[idx]
        
    def updateV(self, vGlobal):
        """
        update current value of velocity
        Input:
            vGlobal: global vector of velocity
        """
        if self.v_ is None:
            return
        if vGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                self.v_[i] = vGlobal[idx]
                
    def updateA(self, aGlobal):
        """
        update current value of acceleration
        Input:
            aGlobal: global vector of acceleration
        """
        if self.a_ is None:
            return
        if aGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                self.a_[i] = aGlobal[idx]
        
    def getTimeOrder(self):
        """
        get time order of this node
        """
        return self.timeOrder
        
    def setConstraint(self, free, val, dof):
        """
        Set the constraint for specific dof
        Input:
            free: True if this dof is free
            val:  Value of this dof
            dof:  index of dof
        """
        if dof < 0 or dof >= self.Ndof:
            raise WrongDofIndexException
        self.u_[dof] = val
        self.freedom[dof] = free
        if not free:
            if math.fabs(val) > 1.0e-14:
                self.NonHomogeneousDirichlet = True
                
    def hasNonHomogeneousDirichlet(self):
        """
        Return True if node has nonhomogeneous Dirichlet boundary conditions
        """
        return self.NonHomogeneousDirichlet
        
    def assembleGlobalDirichlet(self, vGlobalD):
        """
        assemble nonhomogeneous Dirichlet boundary conditions to 
        global displacement vector
        Input:
            vGlobalD: global displacement vector
        """
        if vGlobalD is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx < -1:
                vGlobalD[-idx-1] = self.u_[i]
                
    def assembleU(self, uGlobal):
        """
        assemble diplacement of this node to global displacement vector
        Input:
            uGlobal: global displacement vector
        """
        if uGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                uGlobal[idx] = self.u_[i]
                
    def assembleV(self, vGlobal):
        """
        assemble velocity of this node to global velocity vector
        Input:
            uGlobal: global velocity vector
        """
        if vGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                vGlobal[idx] = self.v_[i]
                
    def assembleA(self, aGlobal):
        """
        assemble acceleration of this node to global acceleration vector
        Input:
            uGlobal: global acceleration vector
        """
        if aGlobal is None:
            return
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                aGlobal[idx] = self.a_[i]
        
    def setLoad(self, load, dof):
        """
        Set the load or source for specific dof
        Input:
            load: value or function of load or source
                  if load is a function, it should has form f(t)
                  with t is time
            dof: index of dof
        """
        if dof < 0 or dof >= self.Ndof:
            raise WrongDofIndexException
        if callable(load):
            self.callPointLoad = True
        self.load[dof] = load
        
    def getPointLoad(self, dof, t = 0):
        """
        Get point load at degree of freedom dof
        Input:
            dof: degree of freedom
            t: time, default is t=0
        Return value of load
        """
        if self.callPointLoad:
            return self.load[dof](t)
        return self.load[dof]
        
    def getPointLoadToGlobal(self, Load, t= 0):
        """
        Get point load to global load vector
        Input:
            Load: global load vector
            t: time, default is t=0
        """
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                Load[idx] = self.getPointLoad(i,t)
        
    def hasPointLoad(self):
        """
        If this node has point load, return True
        else, return False
        """
        for i in range(self.Ndof):
            if callable(self.load[i]):
                return True
            if math.fabs(self.load[i]) > 1.0e-14:
                return True
        return False
        
    def timeDependentLoad(self):
        """
        Return True if there is time dependent load, False otherwise
        """
        return any(callable(ld) for ld in self.load)
        
    def addLoadTo(self, vGlobal):
        """
        add point load or source to global load vector
        Input: 
            vGlobal: global load vector
        """
        for i in range(self.Ndof):
            idx = self.ID[i]
            if idx >= 0:
                vGlobal[idx] += self.load[i]
        
    def setID(self, cnt, cntd):
        """
        Set the indices ID, return the next counter
        Input:
              cnt: the counter, current equation index
              cntd: the counter of contrainted dofs
        Return: (cnt, cntd) counter cnt and cntd after the IDs have been set
        """
        for dof in range(self.Ndof):
            if self.freedom[dof]:
                self.ID[dof] = cnt
                cnt += 1
            elif math.fabs(self.u_[dof]) > 1.0e-14:
                self.ID[dof] = cntd
                cntd -= 1
        return cnt, cntd

# End of Node class definition    

class WrongDofIndexException(Exception):
    """
    Exception for wrong degree of freedom index
    """
    pass