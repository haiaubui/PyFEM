# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:25:38 2017

@author: haiau
"""

import numpy as np
import pylab as pl
import Element.FEMBoundary as FB
from Element.FEMElement import OutsideElement
from Mesh.MeshGenerator import GeneralNode as GNode
from Element.PolarElement import PolarNode
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
        self.polar = False
        self.current = False
        
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
        if Element.current:
            self.current = True
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
                if e.current:
                    self.current = True
        
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
        
        for n in self.Nodes:
            if (n.friendNode is not None) and (n.friendDOF > -1):
                n.ID[n.friendDOF] = n.friendNode.ID[n.friendDOF]
        
    def plot(self, fig = None, col = '-b', fill_mat = False, e_number = False,\
    n_number = False, body_load=False, deformed = False, deformed_factor=1.0):
        """
        plot the mesh
        """
        dfact = deformed_factor
        patches = []
        colors = []
        max_mat = 0
        has_2dim = False
        for i,e in enumerate(self.Elements):
            if e_number:
                fig,nodes = e.plot(fig, col, number = i, deformed = deformed,\
                                   deformed_factor=dfact)
            else:
                fig,nodes = e.plot(fig, col, deformed = deformed, \
                                   deformed_factor=dfact)
            if (fill_mat and e.Ndime == 2) or (body_load and e.hasBodyLoad()):
                has_2dim = True
                if self.polar:
                    verts = \
                    [np.array([n.getX(e.loop[i])[1],n.getX(e.loop[i])[0]])\
                    for i,n in enumerate(nodes)]
                else:
                    if not deformed:
                        verts = [n.getX() for n in nodes]
                    else:
                        verts = [n.getX() + dfact*n.getU()[0:2] for n in nodes]
                patches.append(PlotPoly(verts,closed=True))
                if fill_mat:
                    m_id = e.getMaterial().getID()
                if body_load and e.hasBodyLoad():
                    m_id = e.id_bodyLoad + 0.5
                colors.append(m_id)
                if m_id > max_mat:
                    max_mat = m_id
        
        if n_number:
            for i,node in enumerate(self.Nodes):
                if self.polar:
                    pl.text(node.getX()[1],node.getX()[0],str(i)) 
                else:
                    pl.text(node.getX()[0],node.getX()[1],str(i))        
                
        if fill_mat and has_2dim:
            collection = PatchCollection(patches)
            jet = pl.get_cmap('jet')
            cNorm  = Normalize(vmin=0, vmax=max_mat)
            collection.set_color(jet(cNorm(colors)))
            if self.polar:
                ax = fig.add_subplot(111,projection='polar')
            else:
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
    
    def meshgrid(self, boders, res, min_res):
        """
        Create grid from within boders.
        Return list of grid nodes that can be used for post processing.
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        """
        if isinstance(res, (list,np.ndarray,tuple)):
            res0 = res[0]
            res1 = res[1]
        else:
            res0 = res
            res1 = res
        anchors = []
        for n in self.Nodes:
            if n.isInside(boders):
                anchors.append(n)
                
        x = np.arange(boders[0],boders[1]+res0*0.5,res0).tolist()
        y = np.arange(boders[2],boders[3]+res1*0.5,res1).tolist()
        for n in anchors:
            x_ = n.getX()
            x.append(x_[0])
            y.append(x_[1])
            
        x.sort()
        y.sort()
                
        refine_list(x, min_res)
        refine_list(y, min_res)
                
        X,Y = np.meshgrid(x,y)
        return nodes_from_meshgrid(X, Y, self.polar),X,Y
        
    def getValue(self, x, val='u'):
        """
        Get value at position x
        Raise OutsideMesh if x is not lying inside mesh
        """
        for i,e in enumerate(self.Elements):
            try:
                xi = e.getXi(x)
                return e.postCalculate(xi, val)
                #return e.getValFromX(x, val)
            except OutsideElement:
                continue
        raise OutsideMesh
        
    def meshgridValue(self, boders, res, min_res, val='u', idof = 0, idir = 0):
        """
        Get mesh grid and value within boders
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        idir: direction of gradient
        """
        nodes,X,Y = self.meshgrid(boders,res,min_res)
        values = np.empty(len(nodes),self.Nodes[0].dtype)
        values.fill(np.nan)
        grad = val == 'gradu'
        for i,n in enumerate(nodes):
            try:
                value = self.getValue(n.getX(),val)               
            except OutsideMesh:
                #print(n)
                continue
            if grad:
                values[i] = value[idir][idof]
            else:
                values[i] = value[idof]
            
        return X,Y,values.reshape(X.shape)
    
    def meshgridStrain(self, boders, res, min_res, dofx, dofy):
        """
        Get mesh grid and stress within boders
        """
        nodes,X,Y = self.meshgrid(boders,res,min_res)
        ndim = nodes[0].Ndim
        values = np.empty((len(nodes),ndim,ndim))
        for i,n in enumerate(nodes):
            values[i,:,:].fill(np.nan)
            try:
                value = self.getValue(n.getX(),'gradu')
            except OutsideMesh:
                continue
            valuex = value[:,(dofx,dofy)]
            strain = 0.5*(valuex + valuex.T) + np.dot(valuex,valuex.T) 
            values[i,:,:] = strain
            
        return X,Y,values[:,0,0].reshape(X.shape),\
            values[:,0,1].reshape(X.shape),values[:,1,1].reshape(X.shape)    
        
    def meshValue(self, meshx, meshy, val = 'u', idof=0, idir = 0):
        """
        Get values at mesh defined by rectangular mesh meshy and meshy
        meshx: array, mesh points in x direction
        meshy: array, mesh points in y direction
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        idir: direction of gradient
        """
        X,Y = np.meshgrid(meshx,meshy)
        nodes = nodes_from_meshgrid(X, Y)
        values = np.empty(len(nodes),self.Nodes[0].dtype)
        values.fill(np.nan)
        grad = val == 'gradu'
        for i,n in enumerate(nodes):
            try:
                value = self.getValue(n.getX(),val)[idof]                
            except OutsideMesh:
                print(n)
                continue
            if grad:
                values[i] = value[idir]
            else:
                values[i] = value
            
        return X,Y,values.reshape(X.shape)
        
    def contour(self, boders, res, min_res, fig = None, val='u', idof = 0):
        """
        Create countour plot within boders.
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        """
        X,Y,Z = self.meshgridValue(boders,res,min_res,val,idof)
            
        if fig is None:
            fig = pl.figure()
        pl.contour(X,Y,Z)
        pl.gca().set_aspect('equal', adjustable='box')
        return fig
        
    def contourf(self, boders, res, min_res, fig = None, val='u', idof = 0):
        """
        Create a contour filled with color within boders.
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        """
        X,Y,Z = self.meshgridValue(boders,res,min_res,val,idof)
            
        if fig is None:
            fig = pl.figure()
        
        pl.contourf(X,Y,Z)
        pl.gca().set_aspect('equal', adjustable='box')
        return fig

    
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
    
    def meshgrid(self, boders, res, min_res):
        """
        Create grid from within boders.
        Return list of grid nodes that can be used for post processing.
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        """
        if isinstance(res, (list,np.ndarray,tuple)):
            res0 = res[0]
            res1 = res[1]
        else:
            res0 = res
            res1 = res
        anchors = []
        for n in self.Nodes:
            if n.isInside(boders):
                anchors.append(n)
                
        self.nanPoints = []
#        for e in self.BoundaryElements:
#            for n in e.Nodes:
#                if not n.isInside(boders):
#                    continue
#                nx = n.copyToPosition(n.getX()+1.0e-8*e.normv)
#                if nx not in anchors:
#                    anchors.append(nx)
#                    self.nanPoints.append(nx)
                
        x = np.arange(boders[0],boders[1]+res0*0.5,res0).tolist()
        y = np.arange(boders[2],boders[3]+res1*0.5,res1).tolist()
        for n in anchors:
            x_ = n.getX()
            x.append(x_[0])
            y.append(x_[1])
            
        x.sort()
        y.sort()
                
        refine_list(x, min_res)
        refine_list(y, min_res)
                
        X,Y = np.meshgrid(x,y)
        nodes = nodes_from_meshgrid(X, Y, self.polar)
#        for e in self.BoundaryElements:
#            for n in nodes:
#                try:
#                    if e.distanceToPoint(n.getX())<1.0e-7:
#                        if n not in self.nanPoints:
#                            self.nanPoints.append(n)
#                except FB.InsideElement:
#                    continue
        return nodes,X,Y
    
        
    def plot(self, fig = None, col = '-b',colb = '-r', fill_mat = False, e_number = False,\
    n_number = False, deformed = False, deformed_factor=1.0):
        """
        plot the mesh
        """
        dfact = deformed_factor
        patches = []
        colors = []
        max_mat = 0
        for i,e in enumerate(self.Elements):
            if isinstance(e, FB.StandardBoundary):
                col1 = colb
            else:
                col1 = col
            if e_number:
                fig,nodes = e.plot(fig, col1, number = i, deformed = deformed,\
                                   deformed_factor=dfact)
            else:
                fig,nodes = e.plot(fig, col1, deformed = deformed,\
                                   deformed_factor=dfact)
            if fill_mat and e.Ndime == 2:
                if not deformed:
                    verts = [n.getX() for n in nodes]
                else:
                    verts = [n.getX() + dfact*n.getU()[0:2] for n in nodes]
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
#            fig.colorbar(collection, ax = ax)
            
        pl.gca().set_aspect('equal', adjustable='box')
        return fig
        
    def plotBoundary(self, fig = None, col = '-r'):
        for i,e in enumerate(self.BoundaryElements):
            fig,nodes = e.plot(fig, col)
            
        pl.gca().set_aspect('equal', adjustable='box')
        return fig
        
    def getValue(self, x, val='u',intDat = None):
        """
        Get value at position x
        If val=='u' and x is outside mesh, calculate value from boundary.
        Raise OutsideMesh otherwise
        """
        for i,e in enumerate(self.Elements):
            if i >= (self.Ne-self.NBe):
                continue
            try:
                xi = e.getXi(x)
                return e.postCalculate(xi, val)
            except OutsideElement:
                continue
        if val != 'u':
            raise OutsideMesh
            
        res = np.zeros(self.Nodes[0].Ndof,self.Nodes[0].dtype)
        for e in self.BoundaryElements:
            res += e.postCalculateX(x,intDat)
            
        return res
    
    def meshValue(self, meshx, meshy, val = 'u', idof=0, idir = 0):
        """
        Get values at mesh defined by rectangular mesh meshy and meshy
        meshx: array, mesh points in x direction
        meshy: array, mesh points in y direction
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        idir: direction of gradient
        """
        X,Y = np.meshgrid(meshx,meshy)
        nodes = nodes_from_meshgrid(X, Y)
        values = np.empty(len(nodes),self.Nodes[0].dtype)
        values.fill(np.nan)
        grad = val == 'gradu'
        for i,n in enumerate(nodes):
            try:
                if n in self.nanPoints and val == 'u':
                    value = np.nan
                else:
                    value = self.getValue(n.getX(),val)[idof]                
            except OutsideMesh:
                print(n)
                continue
            if grad:
                values[i] = value[idir]
            else:
                values[i] = value
            
        return X,Y,values.reshape(X.shape)
    
    def meshgridValue(self, boders, res, min_res, val='u', idof = 0, idir = 0):
        """
        Get mesh grid and value within boders
        boders = [left, right, under, upper]
        res: resolution of grid, maximum distance in grid
        min_res: minimum distance between points in grid
        val: value to be plotted, 'u', 'v', 'a', or 'gradu'
        idof: index of degree of freedom
        idir: direction of gradient
        """
        nodes,X,Y = self.meshgrid(boders,res,min_res)
        values = np.empty(len(nodes),self.Nodes[0].dtype)
        values.fill(np.nan)
        dummyval = np.zeros(self.Nodes[0].Ndof)
        grad = val == 'gradu'
        print('number of points: '+str(len(nodes)))
        print('finished creating grid, calculating value...')
        for i,n in enumerate(nodes):
            try:
                if n in self.nanPoints and val == 'u':
                    dummyval[idof] = np.nan
                    value = dummyval
                else:
                    value = self.getValue(n.getX(),val)               
            except OutsideMesh:
                #print(n)
                continue
            if grad:
                values[i] = value[idir][idof]
            else:
                values[i] = value[idof]
            
        return X,Y,values.reshape(X.shape)
        
class MeshWithGapElement(Mesh):
    """
    Mesh with gap elements
    """
    def __init__(self, nHarmonic = 10):
        """
        Initialize mesh with gap elements
        Beside element's list, we need another list for boundary elements
        """
        Mesh.__init__(self)
        self.GapElements = []
        self.NBe = 0
        self.nHarmonic = nHarmonic
        self.polar = True
        
    def addGapElement(self, Element):
        """
        add boundary element to mesh
        Notice that Element is also added to Elements list
        Input:
            Element: element to be added
        """
        self.addElement(Element)
        try:
            self.GapElements.append(Element)
        except Exception:
            self.GapElements = []
            self.GapElements.append(Element)
        self.NBe += 1
        
    def addGapElements(self, elements):
        try:
            for e in elements:
                self.addGapElement(e)
        except TypeError:
            self.addGapElement(elements)
    
    def setNumberHarmonic(self, nHarmonic):
        """
        set number of harmonics
        """
        self.nHarmonic = nHarmonic
        
    def setRadiusRatio(self, rRatio):
        """
        set ratio between inner and outer radius
        """
        self.rRatio = rRatio
        
    def setMaterialConstant(self, mconst):
        """
        set material constant
        """
        self.matConst = mconst
        
    def generateID(self):
        if self.Nodes is None or len(self.Nodes) == 0:
            raise EmptyMesh
        cntu = 0
        cntd = -1
        for i in range(self.Nnod):
            cntu, cntd = self.Nodes[i].setID(cntu,cntd)
        
        if self.nHarmonic > 0:
            self.harmonicsIDa = []
            self.harmonicsIDb = []
            self.harmonicsIDc = []
            self.harmonicsIDd = []
            for i in range(1,self.nHarmonic+1):
                self.harmonicsIDa.append(cntu)
                cntu += 1
                self.harmonicsIDb.append(cntu)
                cntu += 1
                self.harmonicsIDc.append(cntu)
                cntu += 1
                self.harmonicsIDd.append(cntu)
                cntu += 1
        
        self.Neq = cntu
        self.NeqD = -cntd - 1
        
    def calculateHarmonicsMatrix(self, data):
        vGlob = data.KtL
        for i in range(self.nHarmonic):
            vGlob[self.harmonicsIDa[i],self.harmonicsIDa[i]]=-(i+1)*np.pi*\
            self.matConst*(1.0-self.rRatio**(2*(i+1)))
            vGlob[self.harmonicsIDb[i],self.harmonicsIDb[i]]=-(i+1)*np.pi*\
            self.matConst*(1.0-self.rRatio**(2*(i+1)))
            vGlob[self.harmonicsIDc[i],self.harmonicsIDc[i]]=-(i+1)*np.pi*\
            self.matConst*(1.0-self.rRatio**(2*(i+1)))
            vGlob[self.harmonicsIDd[i],self.harmonicsIDd[i]]=-(i+1)*np.pi*\
            self.matConst*(1.0-self.rRatio**(2*(i+1)))
            

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

def refine_list(x, min_res):
    """
    delete elements with distances smaller than min_res
    """
    i = 0
    a = len(x)
    while i < a-1:
        if x[i+1]-x[i] < min_res:
            x.pop(i+1)
            a -= 1
        else:
            i += 1
            
def nodes_from_meshgrid(X, Y, polar=False):
    """
    create list of nodes of meshgrid X,Y from numpy function meshgrid
    """
    nodes = []
    if not polar:
        for i,y in enumerate(Y):
            for j,yi in enumerate(y):
                nodes.append(GNode([X[i][j],yi],2))
    else:
        for i,y in enumerate(Y):
            for j,yi in enumerate(y):
                nodes.append(PolarNode([X[i][j],yi],2))
            
    return nodes
    
def findNodeNearX(Nodes, x):
    """
    find the nearest node in Nodes to coordinate x
    """
    if Nodes is None:
        return None, -1
    min_dist = 1.0e15
    min_i = -1
    for i,n in enumerate(Nodes):
        dist = np.linalg.norm(n.getX()-x)
        if dist < min_dist:
            min_dist = dist
            min_i = i
            
    return Nodes[min_i], min_i

class EmptyMesh(Exception):
    """
    Exception for empty mesh (mesh without nodes)
    """
    pass

class OutsideMesh(Exception):
    """
    Exception in case of a point is outside mesh
    """
    pass
       