# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:04:56 2017

@author: haiau
"""

import warnings
import math
import numpy as np
import pylab as pl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as PlotPoly
from matplotlib.colors import Normalize

class GeneralNode(object):
    """
    General node includes only coordinates of node
    """
    def __init__(self, X, ndim):
        """
        Initialized generalized Node
        Input:
            X: coordinates of node
            ndim: number of dimension
        """
        assert ndim > 0, 'Number of dimensions must be a positive number'
        self.Ndim = ndim
        assert len(X) >= ndim, 'There is not enough coordinate'
        self.X_ = np.array(X[0:self.Ndim])
        
    def copyToPosition(self, X):
        """
        Copy this node to some other position X
        Return node of the same type
        """
        ndim = self.Ndim
        n = GeneralNode(X, ndim)
        if self == n:
            raise DuplicatedNode
            
        return n
    
    def getNdim(self):
        """
        Return number of dimensions of this node
        """
        return self.Ndim
        
    def getX(self):
        """
        Return the coordinates of this node, numpy array
        """
        return self.X_
        
    def __eq__(self, node):
        """
        Compare this node to other node
        Two nodes are equal if they have same coordinates and number of dims
        If two nodes have different number of dimensions,
        raise NodesNotSameDimension
        Notice: two nodes a,b will be considered to be equal
        if |a-b|<1.0e-14
        """
        if self.Ndim != node.getNdim():
            raise NodesNotSameDimension
        
        return np.linalg.norm(self.X_ - node.getX()) < 1.0e-14
        
    def __str__(self):
        """
        Display node
        """
        return str(self.X_.tolist())
        

class Edge(object):
    """
    Edge class includes two nodes
    """
    def __init__(self, Node1, Node2):
        """
        Initialize an edge
        Input:
            Node1, Node2: two nodes of an edge
        Raise NodesOppositeOrder if two nodes are the same
        """
        assert Node1 is not None and Node2 is not None,\
        'cannot assign None as a node'
        if Node1 == Node2:
            raise NullEdge
        self.Nodes = [Node1, Node2]
        self.Ndim = Node1.getNdim()
        self.Ndiv = 0
        self.Wb = 1.0
        self.Wm = 1.0
        self.We = 1.0
        
        
        # edges and nodes of divisions
        self.divEdges = None
        self.divNodes = None
        
    def __eq__(self, edge):
        """
        Compare two edges
        Two edges will be considered equal if they have equal nodes
        Raise NodesOppositeOrder if nodes are equal but in opposite order
        Notice: this method does not change any other members of both objects
        """
        nodes = edge.getNodes()
        if nodes[1] == self.Nodes[0] and nodes[0] == self.Nodes[1]:
            raise NodesOppositeOrder
        return nodes[0] == self.Nodes[0] and nodes[1] == self.Nodes[1]
        
    def getNdiv(self):
        """
        Return number of node divisions in this edge
        """
        return self.Ndiv+1
        
    def getNodes(self):
        """
        get two nodes of an edge
        """
        return self.Nodes
        
    def getCenter(self):
        """
        get center coordinate of an edge
        Return a numpy array
        """
        x1 = self.Nodes[0].getX()
        x2 = self.Nodes[1].getX()
        return (x1+x2)*0.5
        
    def setNode(self, node, pos):
        """
        pos = 0: set first node of edge
              1:     second
        """
        if pos == 0:
            self.Nodes[0] = node
        elif pos == 1:
            self.Nodes[1] = node
        
    def switchNodes(self):
        """
        switch order of nodes
        """
        self.Nodes[0],self.Nodes[1] = self.Nodes[1],self.Nodes[0]
        
    def setNumberDivision(self, ndiv):
        """
        set number of division in this edge
        """
        if ndiv > 0:
            self.Ndiv = ndiv-1
            
    def setWeights(self, wb, wm, we, balance_mid = True):
        """
        set the weight at the begining, middle, end of an edge
        Input:
            wb: weight in the begining of edge
            wm:               middle
            we:               end
            balance_mid:True: set middle edge weight to (wb + we)/2
                        False: set middle edge weight to wm
        """
        if wb > 0:
            self.Wb = wb
        if wm > 0:
            if balance_mid:
                self.Wm = (wb+we)*0.5
            else:
                self.Wm = wm
        if we > 0:
            self.We = we
            
    def getEquation(self):
        """
        get three factor of equation a*x + b*y + c = 0
        return a,b,c
        """
        return straightLineEquation(self.Nodes[0],self.Nodes[1])
        
    def getNormalVector(self, point):
        """
        Get vector that has origin at point and normal to this edge and point
        to this edge
        """
        x = normalVectorEdge(self)
        a1,b1,c1 = x[1],-x[0],x[0]*point[1]-x[1]*point[0]
        a2,b2,c2 = self.getEquation()
        c1,c2 = -c1,-c2
        da = a1*b2 - a2*b1
        if math.fabs(da) < 1.0e-14:
            raise NotIntersect
        xi = np.array([(c1*b2-c2*b1)/da,(c2*a1-c1*a2)/da])
        nv = xi - point
        nnv = np.linalg.norm(nv)
        return np.array([nv[0]/nnv,nv[1]/nnv])
        
        
    def divideSections(self, ndiv):
        Npoint = ndiv - 1
        assert Npoint >= 0, 'Number of division points added must be positive'
        if Npoint == 0:
            self.divNodes = self.Nodes
            self.divEdges = self
            return
        self.Ndiv = Npoint
        delta = 1.0/(Npoint+1)
        tn = [(i+1)*delta for i in range(Npoint)]
        a1 = -(self.Wm - self.Wb)
        v1 = 1.0 - a1*0.5
        a2 = -(self.We - self.Wm)
        v2 = 1.0 - a2*0.5
        for i in range(Npoint):
            if tn[i] < 0.5:
                tn[i] = (a1*tn[i]*0.5 + v1)*tn[i]
            else:
                tn[i] = (a2*tn[i]*0.5 + v2)*tn[i]
        X2 = self.Nodes[1].getX()
        X1 = self.Nodes[0].getX()
        self.divNodes = []
        for t in tn:
            self.divNodes.append(\
            self.Nodes[0].copyToPosition(X1 + (X2 - X1)*t))
        self.divEdges = [Edge(self.divNodes[i],self.divNodes[i+1]) \
        for i in range(self.Ndiv - 1)]     
        
    def getDivEdges(self):
        """
        Return divided edges
        """
        return self.divEdges
        
    def getDivNodes(self):
        """
        Return nodes that divide this edge
        """
        return self.divNodes
        
    def extendToQuad(self, d, s):
        """
        Extend edge to quadrilateral
        Input:
            d: vector, direction of extension, numpy array
            s: distance to extend
               s may be an array, then the edge will be extended
               consecutively for each s in array
        Return a quadrilateral for single s or list of quadrilateral for array
        """
        if isinstance(s, (list,tuple,np.ndarray)):
            quads = []
            quads.append(self.extendToQuad(d,s[0]))
            for i in range(1,len(s)):
                quads.append(quads[i-1].getEdges()[2].extendToQuad(d,s[i]))
            return quads
        
        node1 = self.Nodes[0]
        node2 = self.Nodes[1]
        node3 = node2.copyToPosition(self.Nodes[1].getX() + s*d)
        node4 = node2.copyToPosition(self.Nodes[0].getX() + s*d)
        
        return Quadrilateral([node1,node2,node3,node4])
        
    def plot(self, fig = None, col = 'xb-'):
        """
        plot edge
        """
        if fig is None:
            fig = pl.figure()
        X1 = self.Nodes[0].getX()
        X2 = self.Nodes[1].getX()
        pl.plot(np.array([X1[0],X2[0]]),np.array([X1[1],X2[1]]),col)
        return fig
        
    def __str__(self):
        """
        Display the edge
        """
        return str(self.Nodes[0].getX().tolist())+'-->'+\
        str(self.Nodes[1].getX().tolist())
        

class Polygon(object):
    """
    Polygons object
    """
    def __init__(self, Nnod, Nodes = None, nodeOrder = None):
        """
        Initialize polygons object
        Input:
            Nnod: number of node 3 <= Nnod
            Nodes: list of nodes, default is None
            the order follows the cycling rule:
                   *i+2                  i+3--------------*i+2
                   |\                       |              |
                   | \                      |              |
                   |  \                     |              |
                  i*---*i+1                i*--------------*i+1
            nodeOrder: ordering of node, list or array, default is None
                       deprecated: only convex polygons are considered,
                                   therefore nodeOrder is unneccessary
                 
        """
        assert Nnod >= 3, 'Polygon must have at least three vertices'
        if Nodes is not None:
            assert len(Nodes) == Nnod,\
            'this polygon requires '+str(Nnod)+' nodes'
        if nodeOrder is not None :
            assert len(nodeOrder) == Nnod,'invalid node order'
        self.Nnod = Nnod
        self.Nodes = Nodes
        
        # deprecated variable
        self.nodeOrder = nodeOrder
        
        self.Edges = None
        self.Nedge = self.Nnod
        if self.Nodes is not None and self.nodeOrder is not None:
            self.arrangeNodes()
        if self.Nodes is not None and len(self.Nodes) == self.Nnod:
            self.nodeOrder = [i for i in range(self.Nnod)]
            self.arrangeNodes()
            
        self.clw = clockwise(self.Nodes)
            
        # material of polygon
        self.material = None
        
        # external body load of polygon
        self.bodyLoad = None
        
        # edges that have reversed node order
        self.reversed = [False]*Nnod
        
        # Polygons of elements of mesh created
        self.meshPoly = None
        
        # Points of mesh created
        self.points = None
        self.__iter_index = 0
        
    def __iter__(self):
        self.__iter_index = 0
        return self
        
    def __next__(self):
        if self.__iter_index == self.Nnod or self.Nodes is None:
            raise StopIteration
        idx = self.__iter_index
        self.__iter_index += 1
        return self.Nodes[idx]
        
    def __eq__(self, poly):
        """
        Check if two polygons are equal
        """
        if self.Nnod != poly.getNnod():
            return False
        norder = poly.getNodeOrder()
        nodes = poly.getNodes()
        a = (self.Nodes[self.nodeOrder[i]]==nodes[norder[i]]\
        for i in range(self.Nnod))
            
        return all(a)
        
    def isClockwise(self):
        """
        return True if nodes are ordering clocwise
        """
        return self.clw > 0
        
    def setMaterial(self, material):
        """
        Set material of this polygon
        """
        self.material = material
        
    def getMaterial(self):
        """
        Get material of this polygon
        """
        return self.material
        
    def setBodyLoad(self, bodyLoad):
        """
        Set body load to this polygon
        """
        self.bodyLoad = bodyLoad
        
    def getBodyLoad(self):
        """
        Get body load of thif polygon
        """
        return self.bodyLoad
        
    def getNodes(self):
        """
        get nodes of polygon
        """
        return self.Nodes
        
    def getEdges(self):
        """
        get edges of polygon
        """
        return self.Edges
        
    def getNnod(self):
        """
        get number of nodes in polygon
        """
        return self.Nnod
        
    def getNodeOrder(self):
        """
        get order of nodes in polygon
        """
        return self.nodeOrder
        
    def switchEdge(self, iedge):
        """
        switch node order of an edge
        """
        self.reversed[iedge] = not self.reversed[iedge]
        
    def addNode(self, node):
        """
        add node to polygon
        Raise DuplicatedNodes if node is already in polygon
        """
        assert node is not None, 'Cannot add None as node'
        if self.Nodes is None:
            self.Nodes = [node]
            return
        assert len(self.Nodes) < self.Nnod, 'Cannot add more node to polygon'
        if any(node == nodei for nodei in self.Nodes):
            raise DuplicatedNode
            
        self.Nodes.append(node)
        
    def setWeightEdge(self, wb, wm, we, iedge, balance_mid = True):
        """
        set division weight for a specific edge
        """
        assert iedge >= 0 and iedge < 4, 'false edge index'
        assert self.Edges is not None, 'No edges created'
        if self.reversed[iedge]:
            wb,we = we,wb
        self.Edges[iedge].setWeights(wb, wm, we, balance_mid)
        
    def arrangeNodes(self, nodeOrder = None):
        """
        arrange Nodes following nodeOrder
        the edges will follow the node order. Therefore, two polygon with
        two different node order are different
        """
        assert self.nodeOrder is None or nodeOrder is None,\
        'No node ordering is specified'
        if self.nodeOrder is None:
            self.nodeOrder = nodeOrder
            if self.nodeOrder is None:
                self.nodeOrder = [i for i in range(self.Nnod)]
            
        assert len(self.nodeOrder) == self.Nnod, 'invalid node order'
        assert all(i < self.Nnod and i >= 0 for i in self.nodeOrder),\
        'invalid node order'
        assert all( i in self.nodeOrder for i in range(self.Nnod)),\
        'invalid node order'
        
        self.Nodes = [self.Nodes[i] for i in self.nodeOrder]
        self.Edges = [Edge(self.Nodes[i],self.Nodes[(i+1)%self.Nnod])\
        for i in range(self.Nnod)]
            
    def structuredMesh(self, elm_type = 'Quad'):
        """
        Mesh the polygon with structured mesh
        """
        return None
        
            
    def plot(self, fig = None, col = 'xb-'):
        """
        Plot the polygon
        """
        for e in self.Edges:
            fig = e.plot(fig, col)
            
        return fig
        
    def getMeshPolygons(self):
        """
        Return the quad mesh of this polygon
        if polygon has not been meshed yet, it will be meshed with structure
        mesh
        """
        if self.meshPoly is None:
            return self.structuredMesh()
            
        return self.meshPoly
        
    def getMeshPoints(self):
        """
        Return all the points of mesh generated
        """
        return self.points
        
    def clearMesh(self):
        """
        Clear created mesh
        """
        self.meshPoly = None
    
        
    def plotMesh(self, fig = None, col = 'xb-'):
        """
        Plot the mesh created
        """
        return fig
        
    def __str__(self):
        """
        Display the polygon
        """
        self.arrangeNodes()
        s = ''
        for no in self.nodeOrder:
            s += str(self.Nodes[no])+'-->'
        s += str(self.Nodes[self.nodeOrder[0]])
        return s
        
class Geometry(object):
    """
    Geometry class contains all nodes, edges and polygons
    """
    def __init__(self):
        """
        Initialize geometry
        """
        self.Nnod = 0
        self.Nedge = 0
        self.Npoly = 0
        self.Nodes = None
        self.Edges = None
        self.Poly = None
        self.bndMesh = None
        
    def getNodes(self):
        """
        Return nodes of Geometry
        """
        return self.Nodes
        
    def getNnod(self):
        """
        Return number of nodes
        """
        return self.Nnod
        
    def getEdges(self):
        """
        Return edges of Geometry
        """
        return self.Edges
        
    def getNedge(self):
        """
        Return number of edges
        """
        return self.Nedge
        
    def getNpoly(self):
        """
        Return number of polygons
        """
        return self.Npoly
        
    def getPolygons(self):
        """
        Return polygons of Geometry
        """
        return self.Poly
        
    def __contains__(self, obj):
        """
        Check if and object is in geometry
        """
        if isinstance(obj, GeneralNode):
            return obj in self.Nodes
            
        if isinstance(obj, Edge):
            try:
                return obj in self.Edges
            except NodesOppositeOrder:
                warnmess = 'Edge '+str(obj)+\
                ' has one clone edge in geometry,\
                but the nodes are in opposite order'
                warnings.warn(warnmess)
                return True
            
        if isinstance(obj, Polygon):
            return obj in self.Poly
            
    def addNode(self, node):
        """
        Add node to this geometry
        """
        assert node is not None, 'Cannot add None as node'
        if self.Nodes is None:
            self.Nodes = [node]
            self.Nnod = 1
            return
        if any(node == nodei for nodei in self.Nodes):
            warnmess = 'Node '+str(node)+\
            ' is dupplicated with one node in geometry. Node is not added.'
            warnings.warn(warnmess)
            
        self.Nodes.append(node)
        
    def addNodes(self, nodes):
        """
        Add many nodes (a list or tuple) to this geometry
        """
        if isinstance(nodes, (list,tuple)):
            for node in nodes:
                self.addNode(node)
        
    def addEdge(self, edge):
        """
        Add edge to this geometry
        The nodes of this edge will be added to geometry also, if there is no
        such node in geometry
        """
        assert edge is not None, 'Cannot add None as edge'
        if self.Edges is None:
            self.Edges = [edge]
            self.Nedge = 1
            nodes = edge.getNodes()
            if self.Nnod > 0:
                try:
                    idx = self.Nodes.index(nodes[0])
                    nodes[0] = self.Nodes[idx]
                except ValueError:
                    self.addNode(nodes[0])
                    
                try:
                    idx = self.Nodes.index(nodes[1])
                    nodes[1] = self.Nodes[idx]
                except ValueError:
                    self.addNode(nodes[1])
            else:
                self.addNode(nodes[0])
                self.addNode(nodes[1])
            return
        try:
            if any(edge == edgei for edgei in self.Edges):
                warnmess = 'Edge '+str(edge)+\
                ' is dupplicated with one edge in geometry. Edge is not added.'
                warnings.warn(warnmess)
            else:
                self.Edges.append(edge)
                self.Nedge += 1
                nodes = edge.getNodes()
                if self.Nnod == 0:
                    self.addNode(nodes[0])
                    self.addNode(nodes[1])
                    return
                try:
                    idx = self.Nodes.index(nodes[0])
                    nodes[0] = self.Nodes[idx]
                except ValueError:
                    self.addNode(nodes[0])
                    
                try:
                    idx = self.Nodes.index(nodes[1])
                    nodes[1] = self.Nodes[idx]
                except ValueError:
                    self.addNode(nodes[1])
                return
        except NodesOppositeOrder:
            warnmess = 'Edge '+str(edge)+\
            ' has one clone edge in geometry,\
            but the nodes are in opposite order. The edge is not added.'
            warnings.warn(warnmess)
            
    def addEdges(self, edges):
        """
        Add many edges (list or tuple) into geometry
        See addEdge for more information
        """
        if isinstance(edges, (list, tuple)):
            for e in edges:
                self.addEdge(e)
            
    def addPolygon(self, poly):
        """
        Add polygon to this geometry
        All nodes and edges of this polygon will be added to geometry also, if
        they are not available there.
        """
        assert poly is not None, 'Cannot add None as polygon'
        poly.arrangeNodes()
        if self.Poly is None:
            self.Poly = [poly]
            self.Npoly = 1
            edges = poly.getEdges()
            if self.Nedge > 0:
                for i in range(len(edges)):
                    try:
                        idx = self.Edges.index(edges[i])
                        edges[i] = self.Edges[idx]
                        warnings.warn('Replace edge in geometry')
                    except ValueError:
                        self.addEdge(edges[i])
                    except NodesOppositeOrder:
                        edges[i].switchNodes()
                        idx = self.Edges.index(edges[i])
                        edges[i] = self.Edges[idx]
                        poly.switchEdge(i)
            else:
                for e in edges:
                    self.addEdge(e)
        else:
            if any(poly == polyi for polyi in self.Poly):
                warnmess = 'Edge '+str(poly)+\
                ' is dupplicated with one polygon in geometry.\
                Polygon is not added.'
                warnings.warn(warnmess)
            else:
                self.Poly.append(poly)
                self.Npoly += 1
                edges = poly.getEdges()
                if self.Nedge > 0 and edges is not None:
                    for i in range(len(edges)):
                        try:
                            idx = self.Edges.index(edges[i])
                            edges[i] = self.Edges[idx]
                            warnings.warn('Replace edge in geometry')
                        except ValueError:
                            self.addEdge(edges[i])
                        except NodesOppositeOrder:
                            edges[i].switchNodes()
                            idx = self.Edges.index(edges[i])
                            edges[i] = self.Edges[idx]
                            poly.switchEdge(i)
                else:
                    if edges is not None:
                        for e in edges:
                            self.addEdge(e)
                            
    def addPolygons(self, polys):
        """
        Add many polygons (list or tuple) to geometry
        See addPolygon for more information
        """
        if isinstance(polys, (list, tuple)):
            for p in polys:
                self.addPolygon(p)
                            
    def getMaterials(self):
        """
        Return the list of all materials in this geometry
        """
        mats = []
        for p in self.Poly:
            m = p.getMaterial()
            if m not in mats and m is not None:
                mats.append(p.getMaterial())
        return mats
                            
    def meshBoundary(self, matExclude = None):
        """
        Generate the boundary mesh
        """
        # find the material list
        mats = self.getMaterials()
        if matExclude is not None:
            for m in matExclude:
                try:
                    idx = mats.index[m]
                    del mats[idx]
                except ValueError:
                    pass
                
        # divide polygons into materials
        polymat = [[] for i in range(len(mats))]
        for p in self.Poly:
            try:
                idx = mats.index(p.getMaterial())
                polymat[idx].append(p)
            except ValueError:
                pass
                
        # create list of edge that cover materials
        coveredges = []
        normvec = []
        for m in polymat:
            for p in m:
                edges = p.getEdges()
                cp = centroidPolygon(p)
                for e in edges:
                    try:
                        if e not in coveredges:
                            coveredges.append(e)
                            normvec.append(e.getNormalVector(cp))
                    except NodesOppositeOrder:
                        pass
                    
        # add divisions of each edge to list of elements
        self.bndMesh = []
        self.normBndVec = []
        for i,e in enumerate(coveredges):
            e.divideSections(e.Ndiv+1)
            me = e.getDivEdges()
            try:
                for de in me:
                    try:
                        if de not in self.bndMesh:
                            self.bndMesh.append(de)
                            self.normBndVec.append(normvec[i])
                    except NodesOppositeOrder:
                        pass
            except TypeError:
                try:
                    if me not in self.bndMesh:
                        self.bndMesh.append(me)
                        self.normBndVec.append(normvec[i])
                except NodesOppositeOrder:
                    pass
                
    def getBoundaryMesh(self, nodeFunc, nodeElemFunc, Ndof, nodes = None):
        """
        Creates nodes and list of element nodes from boundary mesh
        Input:
            nodeFunc: function to create node
            nodes: nodes from body mesh, if the nodes in boundary mesh can be
            found in nodes, it will be skipped
            Ndof: number of degree of freedom in each node
            nodeElemFunc: function that create nodes for each element
        Return:
            modified list of nodes created by nodeFunc
            list of list of nodes in boundary elements
            list of norm vector of each element
        Notice:
            if no boundary mesh was created, meshBoundary() will be called
            without excluded materials. It is therefore recommended that the
            meshBoundary() would be called before this method is called
        """
        if self.bndMesh is None:
            self.meshBoundary()
            
        if nodes is None:
            nodes = []
            
        if nodeFunc is None:
            nf = self.getNodes()[0].copyToPosition
            
        elem = []
        for e in self.bndMesh:
            nodesx = nodeElemFunc(e)
            Nenod = len(nodesx)
            elem.append([None]*Nenod)
            for i in range(Nenod):
                try:
                    idx = nodes.index(nodesx[i])
                    elem[-1][i] = nodes[idx]
                except ValueError:
                    if nodeFunc is not None:
                        elem[-1][i] = nodeFunc(nodesx[i].getX(),Ndof)
                    else:
                        elem[-1][i] = nf(nodesx[i].getX())
                    nodes.append(elem[-1][i])
                    
        return nodes, elem, self.normBndVec
        
    def mesh(self, structure = True, elm_type = 'Quad'):
        """
        Mesh the geometry
        Input:
            structure: True: generate structured mesh (default)
                       False: generate unstructured mesh
            elm_type: 'Quad': Quadrilateral element
                      'Tria': Triangular element
        """
        if structure:
            for p in self.Poly:
                p.structuredMesh(elm_type)
                
    def getMesh(self, nodeFunc, nodeElemFunc, Ndof):
        """
        Create nodes and list of element nodes from mesh.
        Input:
            nodeFunc: used for node creation, if nodeFunc is None, copy one
                      node in geometry to new position
            nodeElemFunc: used to create element nodes from a polygon element
            Ndof: number of degree of freedom in each node
        Return: list of nodes created by nodeFunc
                list of list of nodes of each element
                list of material of each element
                list of external body load
        """
        nodes = []
        elem = []
        mat = []
        bdl = []
        if nodeFunc is None:
            nf = self.getNodes()[0].copyToPosition
        for P in self.Poly:
            for p in P.getMeshPolygons():
                nodesx = nodeElemFunc(p)
                #assert len(nodesx) == Nenod, 'nodeElemFunc not compatible'
                Nenod = len(nodesx)
                elem.append([None]*Nenod)
                mat.append(P.getMaterial())
                bdl.append(P.getBodyLoad())
                for i in range(Nenod):
                    try:
                        idx = nodes.index(nodesx[i])
                        elem[-1][i] = nodes[idx]
                    except ValueError:
                        if nodeFunc is not None:
                            elem[-1][i] = nodeFunc(nodesx[i].getX(),Ndof)
                        else:
                            elem[-1][i] = nf(nodesx[i].getX())
                        nodes.append(elem[-1][i])
                        
        return nodes, elem, mat, bdl
    
    def plot(self, fig = None, col = 'xb-', poly_number = False,\
    edge_number = False, fill_mat = False):
        """
        Plot the polygon
        """
        for p in self.Poly:
            fig = p.plot(fig, col)
            
        if poly_number:
            for i,p in enumerate(self.Poly):
                c = centroidPolygon(p)
                pl.text(c[0],c[1],str(i))
            
        if edge_number:
            for i,e in enumerate(self.Edges):
                c = e.getCenter()
                pl.text(c[0],c[1],str(i))
                
        if fill_mat:
            patches = []
            colors = []
            for p in self.Poly:
                if p.getMaterial() is None:
                    continue
                nodes = p.getNodes()
                verts = [n.getX() for n in nodes]
                patches.append(PlotPoly(verts,closed = True))
                colors.append(p.getMaterial().getID())
                
            collection = PatchCollection(patches)
            jet = pl.get_cmap('jet')
            cNorm  = Normalize(vmin=0, vmax=len(self.getMaterials()))
            collection.set_color(jet(cNorm(colors)))
            ax = fig.add_subplot(111)
            ax.add_collection(collection)
            
        pl.gca().set_aspect('equal', adjustable='box')
        return fig                    
                
    def plotMesh(self, fig = None, col = 'xb-', fill_mat = False):
        """
        Plot the mesh created
        """
        for p in self.Poly:
            fig = p.plotMesh(fig,col)
            
        if fill_mat:
            patches = []
            colors = []
            for p in self.Poly:
                if p.getMaterial() is None:
                    continue
                nodes = p.getNodes()
                verts = [n.getX() for n in nodes]
                patches.append(PlotPoly(verts,closed = True))
                colors.append(p.getMaterial().getID())
                
            collection = PatchCollection(patches)
            jet = pl.get_cmap('jet')
            cNorm  = Normalize(vmin=0, vmax=len(self.getMaterials()))
            collection.set_color(jet(cNorm(colors)))
            ax = fig.add_subplot(111)
            ax.add_collection(collection)
            
        pl.gca().set_aspect('equal', adjustable='box')
        return fig

class Quadrilateral(Polygon):
    """
    Quadrilateral, convex polygon with 4 nodes
    """
    def __init__(self, Nodes, nodeOrder = None):
        """
        Initiallize quadrilateral:
        Input:
            Nodes: list of 4 nodes, if list longer than 4 elements, it will
                   take only begining 4 nodes
            nodeOrder: order of nodes
                       deprecated: only convex polygons are considered,
                                   therefore nodeOrder is unneccessary
        """
        Polygon.__init__(self, 4, Nodes, nodeOrder)
        self.meshPoly = None
        self.points = None

    def setDivisionEdge13(self, ndiv):
        """
        set division for first and third edges
        """
        self.Edges[0].divideSections(ndiv)
        self.Edges[2].divideSections(ndiv)
        
    def setDivisionEdge24(self, ndiv):
        """
        set division weight for second and fourth edges
        """
        self.Edges[1].divideSections(ndiv)
        self.Edges[3].divideSections(ndiv)
        
    def setWeightParallelEdges(self, wb, wm, we, iedge, balance_mid = True):
        """
        Set weight of two parallel edges
        """
        if iedge == 0:
            self.setWeightEdge(wb,wm,we,0,balance_mid)
            self.setWeightEdge(we,wm,wb,2,balance_mid)
        elif iedge == 1:
            self.setWeightEdge(wb,wm,we,1,balance_mid)
            self.setWeightEdge(we,wm,wb,3,balance_mid)
        elif iedge == 2:
            self.setWeightEdge(we,wm,wb,0,balance_mid)
            self.setWeightEdge(wb,wm,we,2,balance_mid)
        elif iedge == 3:
            self.setWeightEdge(we,wm,wb,1,balance_mid)
            self.setWeightEdge(wb,wm,we,3,balance_mid)
        else:
            assert 'wrong edge index'
    
    def structuredMesh(self, elm_type = 'Quad'):
        """
        Mesh the quadrilateral with structured mesh
        Input:
            elm_type: type of element
                      'Quad': quadrilateral element
                      'Tria': Triangular element
        Return list of polygons (quadrilateral or triangular) corresponding to
        elements created.
        """
        if elm_type == 'Quad':
            return self.structuredMeshQuad()
        
    def structuredMeshQuad(self):
        """
        Mesh the quadrilateral with structured mesh
        Return list of quadrilaterals corresponding to
        elements created.
        """
        # Check if system is structured mesh possible
        assert self.Edges[0].getNdiv() == self.Edges[2].getNdiv() and \
        self.Edges[1].getNdiv() == self.Edges[3].getNdiv(),\
        'Cannot create structured mesh for this quadrilateral'
        # Create points of intersection inside quadrilateral
        points = [[self.Nodes[0]]]
        nodes1,nodes2 = self.Edges[0].getDivNodes(),self.Edges[1].getDivNodes()
        nodes3,nodes4 = self.Edges[2].getDivNodes(),self.Edges[3].getDivNodes()
        if self.reversed[0] and nodes1 is not None:
            nodes1.reverse()
        if self.reversed[1] and nodes2 is not None:
            nodes2.reverse()
        if self.reversed[2] and nodes3 is not None:
            nodes3.reverse()
        if self.reversed[3] and nodes4 is not None:
            nodes4.reverse()
        ndiv1 = self.Edges[0].getNdiv()-1
        ndiv2 = self.Edges[1].getNdiv()-1
        if nodes1 is not None:
            points[0] += nodes1
        points[0].append(self.Nodes[1])
        for i in range(ndiv2):
            node1 = nodes2[i]
            node2 = nodes4[ndiv2 - i - 1]
            points.append([node2])
            for j in range(ndiv1):
                node3 = nodes1[j]
                node4 = nodes3[ndiv1 - j - 1]
                points[i+1].append(\
                node3.copyToPosition(intersectionLine(node1,node2,node3,node4)))
            points[i+1].append(node1)
        points.append([self.Nodes[3]])
        if nodes3 is not None:
            points[ndiv2+1] += list(reversed(nodes3))
        points[ndiv2+1].append(self.Nodes[2])
        
        self.points = points
        # Create quadrilaterals
        quad = []
        for i in range(ndiv2+1):
            for j in range(ndiv1+1):
                nodes = [points[i][j],points[i][j+1],\
                points[i+1][j+1],points[i+1][j]]
                quad.append(Quadrilateral(nodes,[0,1,2,3]))
                
        self.meshPoly = quad
        return quad
                       
    def plotMesh(self, fig = None, col = 'xb-'):
        """
        Plot the mesh created
        """
        if self.meshPoly is None:
            self.structuredMesh()
        
        for quad in self.meshPoly:
            fig = quad.plot(fig,col)
            
        return fig        
    
def straightLineEquation(Node1, Node2):
    """
    Return parameters a,b,c of a straight line equation a*x + b*y + c = 0
    """        
    X2 = Node1.getX()
    X1 = Node2.getX()
    return X2[1]-X1[1],X1[0]-X2[0],X1[1]*X2[0]-X2[1]*X1[0]

def intersectionLine(Node1, Node2, Node3, Node4):
    """
    Return the coordinate of an intersection bewteen two lines:
    one connects Node1 and Node2, the other connects Node3 and Node4
    If there is no intersection, i.e. parrallel, raise NotIntersect exception
    """       
    a1,b1,c1 = straightLineEquation(Node1, Node2)
    a2,b2,c2 = straightLineEquation(Node3, Node4)
    c1,c2 = -c1,-c2
    da = a1*b2 - a2*b1
    if math.fabs(da) < 1.0e-14:
        raise NotIntersect
    return np.array([(c1*b2-c2*b1)/da,(c2*a1-c1*a2)/da])
    
def nodesQuad9(quad):
    """
    Get nodes of a quadratic quadrilateral element
    The node order is: (1,1)(1,2)(1,3)(2,1)(2,2)(2,3)(3,1)(3,2)(3,3)
    """
    if quad is None:
        return None
    
    quadnod = quad.getNodes()
    if not quad.isClockwise():
        quadnod = quad.getNodes()
    else:
        quadnod = list(reversed(quad.getNodes()))
    assert quad is not None, 'No node in quadrilateral'
    nodes = [quadnod[0]]
    nodes.append(nodes[0].copyToPosition(\
    (quadnod[0].getX() + quadnod[1].getX())*0.5))
    nodes.append(quadnod[1])
    nodes.append(nodes[0].copyToPosition(\
    (quadnod[0].getX() + quadnod[3].getX())*0.5))
    nodes.append(nodes[0].copyToPosition(\
    (quadnod[1].getX() + quadnod[3].getX())*0.5))
    nodes.append(nodes[0].copyToPosition(\
    (quadnod[1].getX() + quadnod[2].getX())*0.5))
    nodes.append(quadnod[3])
    nodes.append(nodes[0].copyToPosition(\
    (quadnod[3].getX() + quadnod[2].getX())*0.5))
    nodes.append(quadnod[2])
    
    return nodes
    
def nodesBoundaryQuad9(bnd):
    """
    Get nodes of boundary of quadratic quadrilateral element
    """
    if bnd is None:
        return None
        
    nodes = bnd.getNodes()
    res = [nodes[0]]
    x = (nodes[0].getX() + nodes[1].getX())*0.5
    res.append(nodes[0].copyToPosition(x))
    res.append(nodes[1])
    
    return res
    
def normalVectorEdge(edge):
    """
    Return normal vector of an edge
    """
    nodes = edge.getNodes()
    A = nodes[1].getX() - nodes[0].getX()
    a,b = A[0],A[1]
    n = np.linalg.norm(A)
    return np.array([b/n,-a/n])

def centroidPolygon(poly):
    """
    Return centroid of a polygon
    """
    nodes = poly.getNodes()
    Nnod = poly.getNnod()
    x = [n.getX()[0] for n in nodes]
    y = [n.getX()[1] for n in nodes]
    a = (x[i]*y[(i+1)%Nnod] - x[(i+1)%Nnod]*y[i] for i in range(Nnod))
    signedA = 0.5*sum(a)*6
    cx = ((x[i]+x[(i+1)%Nnod])*(x[i]*y[(i+1)%Nnod]-x[(i+1)%Nnod]*y[i]) \
    for i in range(Nnod))
    cy = ((y[i]+y[(i+1)%Nnod])*(x[i]*y[(i+1)%Nnod]-x[(i+1)%Nnod]*y[i]) \
    for i in range(Nnod))
    return np.array([sum(cx)/signedA,sum(cy)/signedA])


def clockwise(nodes):
    """
    Check if a list of nodes is clockwise ordered
    Return 1 for clockwise order
          -1 for anti-clockwise order
    Raise NonConvex for neither case is valid
    """
    nn = len(nodes)
    assert all(n.getX().size == 2 for n in nodes),'only for 2 dimensional case'
    for i in range(0,nn-1):
        a,b,c = straightLineEquation(nodes[i],nodes[(i+2)%nn])
        if a < 0.0:
            a = -a
            b = -b
            c = -c
        x1 = nodes[i+1].getX()
        x2 = nodes[(i+3)%nn].getX()
        x3 = nodes[(i+2)%nn].getX()-nodes[i].getX() 
        a1 = a*x1[0] + b*x1[1] + c
        a2 = a*x2[0] + b*x2[1] + c
        if a1 > 0 and a2 < 0:
            if np.all(x3 > 0) or (x3[0] <= 0 and x3[1] >= 0):
                clw = -1
            else:
                clw = 1
        elif a1 < 0 and a2 > 0:
            if np.all(x3 > 0) or (x3[0] <= 0 and x3[1] >= 0):
                clw = 1
            else:
                clw = -1
        try:
            if prevclw != clw:
                raise NonConvex
        except NameError:
            prevclw = clw
            
    return prevclw

class NonConvex(Exception):
    pass
        
class NotIntersect(Exception):
    pass

class NodesNotSameDimension(Exception):
    pass

class NodesOppositeOrder(Exception):
    pass

class NullEdge(Exception):
    pass

class DuplicatedNode(Exception):
    pass

class DuplicatedEdge(Exception):
    pass

if __name__ == "__main__":
    node1 = GeneralNode([0,0],2)
    node2 = GeneralNode([1,0],2)
    node3 = GeneralNode([1,1],2)
    node4 = GeneralNode([0,1],2)
    node5 = GeneralNode([2,0],2)
    node6 = GeneralNode([2,1],2)
    quad = Quadrilateral([node1,node2,node3,node4])
    #quad.getQuadMesh()
    #quad.plotQuadMesh()
    #quadtest = Quadrilateral([node1,node2,node4,node3],[0,1,3,2])
    #print(quadtest == quad)
    quadtest = Quadrilateral([node2,node5,node6,node3])
    #quadtest.addNode(node1)
    #quadtest.addNode(node2)
    #quadtest.addNode(node5)
    #quadtest.addNode(node6)
    geo = Geometry()
    geo.addPolygon(quad)
    geo.addPolygon(quadtest)
    quad.setWeightParallelEdges(2.0,0.7,0.1,0)
    quad.setWeightParallelEdges(2.0,0.7,0.1,1)
    quad.setDivisionEdge13(4)
    quad.setDivisionEdge24(4)
    quadtest.setWeightParallelEdges(2.0,0.7,0.1,0)
    quadtest.setWeightParallelEdges(2.0,0.7,0.1,1)
    quadtest.setDivisionEdge13(4)
    quadtest.setDivisionEdge24(4)
    #quad.getQuadMesh()
    #quad.plotMesh()
    geo.mesh()
    geo.plotMesh()