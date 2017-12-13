# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:25:38 2017

@author: haiau
"""

import numpy as np
import pylab as pl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PlotPoly
from matplotlib.colors import Normalize

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
        nodes = Element.getNodes()
        for i,n in enumerate(nodes):
            try:
                idx = self.Nodes.index(n)
                nodes[i] = self.Nodes[idx]
            except ValueError:
                self.Nodes.append(n)
                self.Nnod += 1
        
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
        if self.Nodes is None or len(self.Nodes) == 0:
            raise EmptyMesh
        cntu = 0
        cntd = -1
        for i in range(self.Nnod):
            cntu, cntd = self.Nodes[i].setID(cntu,cntd)
        
        self.Neq = cntu
        self.NeqD = -cntd - 1
        
    def plot(self, fig = None, col = '-b', fill_mat = False, e_number = False,\
    n_number = False):
        """
        plot the mesh
        """
        patches = []
        colors = []
        max_mat = 0
        for i,e in enumerate(self.Elements):
            if e_number:
                fig,nodes = e.plot(fig, col, number = i)
            else:
                fig,nodes = e.plot(fig, col)
            if fill_mat:
                verts = [n.getX() for n in nodes]
                patches.append(PlotPoly(verts,closed=True))
                m_id = e.getMaterial().getID()
                colors.append(m_id)
                if m_id > max_mat:
                    max_mat = m_id
        
        if n_number:
            for i,node in enumerate(self.Nodes):
                pl.text(node.getX()[0],node.getX()[1],str(i))        
                
        if fill_mat:
            collection = PatchCollection(patches)
            jet = pl.get_cmap('jet')
            cNorm  = Normalize(vmin=0, vmax=max_mat)
            collection.set_color(jet(cNorm(colors)))
            ax = fig.add_subplot(111)
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
            fig.colorbar(collection, ax = ax)
            
        pl.gca().set_aspect('equal', adjustable='box')
        return fig
        
    def findNodeNear(self, x):
        """
        find the nearest node to coordinate x
        """
        if self.Nodes is None:
            return None, -1
        min_dist = 1.0e15
        min_i = -1
        for i,n in enumerate(self.Nodes):
            dist = np.linalg.norm(n.getX()-x)
            if dist < min_dist:
                min_dist = dist
                min_i = i
                
        return self.Nodes[min_i], min_i

    
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
        self.addElement(Element)
        self.BoundaryElements.append(Element)
        self.NBe += 1
        
    def addBoundaryElements(self, elements):
        try:
            for e in elements:
                self.addBoundaryElement(e)
        except TypeError:
            self.addBoundaryElement(elements)
    
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

def get_connections(mesh):
    """
    Get connections of elements.
    Return a list of length Ne, each member of list is a list of node position
    in mesh.Nodes list.
    """
    IT = []
    nodes = mesh.Nodes
    for e in mesh.Elements:
        IT.append([])
        for n in e.Nodes:
            for i in range(mesh.Nnod):
                if n == nodes[i]:
                    IT[-1].append(i+1)
                    
    return IT

def get_connection(mesh, e):
    """
    get connection of element e
    """
    nodes = mesh.Nodes
    IT = []
    for n in e.Nodes:
        for i in range(mesh.Nnod):
            if n == nodes[i]:
                IT.append(i+1)
    return IT

class EmptyMesh(Exception):
    """
    Exception for empty mesh (mesh without nodes)
    """
    pass
        