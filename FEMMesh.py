# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:25:38 2017

@author: haiau
"""

class Mesh(object):
    """
    Mesh for finite element method
    """
    def __init__(self):
        """
        Initialize mesh
        Input:
            Ndim: number of dimensions
        """
        self.Nodes = []
        self.Elements = []
        self.Nnod = 0
        self.Ne = 0
        self.Neq = 0
        self.NeqD = 0
        self.iter_index = 0
        
    def __iter__(self):
        """
        Iterator loops over elements in mesh
        """
        self.iter_index = 0
        return self
        
    def __next__(self):
        """
        next method for iterator
        """
        if self.iter_index == self.Ne:
            raise StopIteration
        i = self.iter_index
        self.iter_index += 1
        return self.Elements[i]
        
    def __str__(self):
        """
        Print information about mesh
        """
        s = 'Number of nodes: '+str(self.Nnod)
        s += '\n'
        s += 'Number of elements: '+str(self.Ne)
        s += '\n'
        s += 'Number of equations: '+str(self.Neq)
        s += '\n'
        s += 'Number of constraints: '+str(self.NeqD)
        return s
        
    def addNode(self, Node):
        """
        add Node to mesh
        """
        self.Nodes.append(Node)
        self.Nnod += 1
        
    def addNodes(self, Nodes):
        """
        Add an array of nodes to mesh
        """
        if not isinstance(Nodes, (list, tuple)):
            self.addNode(Nodes)
        if Nodes is not None:
            for n in Nodes:
                self.addNode(n)
        
    def addElement(self, Element):
        """
        add element to mesh
        """
        self.Elements.append(Element)
        self.Ne += 1
        
    def addElements(self, Elements):
        """
        add an array of elements to mesh
        """
        if not isinstance(Elements, (list, tuple)):
            self.addNode(Elements)
        if Elements is not None:
            for e in Elements:
                self.addElement(e)
        
    def getNodes(self):
        """
        return Nodes of mesh
        """
        return self.Nodes
        
    def getElements(self):
        """
        return elements of mesh
        """
        return self.Elements
        
    def getNnod(self):
        """
        return number of nodes
        """        
        return self.Nnod
        
    def getNe(self):
        """
        return number of elements
        """
        return self.Ne
        
    def getNeq(self):
        """
        return number of equations
        """
        return self.Neq
        
    def getNeqD(self):
        """
        return number of constrained equations
        """
        return self.NeqD
        
    def generateID(self):
        cntu = 0
        cntd = -1
        for i in range(self.Nnod):
            cntu, cntd = self.Nodes[i].setID(cntu,cntd)
        
        self.Neq = cntu
        self.NeqD = -cntd - 1

    
class MeshWithBoundaryElement(Mesh):
    """
    Mesh with boundary elements
    """
    def __init__(self):
        """
        Initialize mesh with boundary elements
        Beside element's list, we need another list for boundary elements
        """
        Mesh.__init__(self)
        self.BoundaryElements = []
        self.NBe = 0
        
    def addBoundaryElement(self, Element):
        """
        add boundary element to mesh
        Notice that Element is also added to Elements list
        Input:
            Element: element to be added
        """
        self.Elements.append(Element)
        self.Ne += 1
        self.BoundaryElements.append(Element)
        self.NBe += 1
    
    def getBoundaryElements(self):
        """
        Return boundary elements
        """
        return self.BoundaryElements
    
    def getNBe(self):
        """
        return number of boundary elements
        """
        return self.NBe
        