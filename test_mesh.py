# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:38:01 2017

@author: haiau
"""

import MeshGenerator as mg
import FEMElement as fe
import FEMNode as fn
import Material as MAT
import numpy as np
import pylab as pl
import math

def test1():
    #node1 = mg.GeneralNode([0,0],2)
    #node2 = mg.GeneralNode([1,0],2)
    #node3 = mg.GeneralNode([1,1],2)
    #node4 = mg.GeneralNode([0,1],2)
    #node5 = mg.GeneralNode([2,0],2)
    #node6 = mg.GeneralNode([2,1],2)
    node1 = fn.Node([0,0],2,2)
    node2 = fn.Node([1,0],2,2)
    node3 = fn.Node([1,1],2,2)
    node4 = fn.Node([0,1],2,2)
    node5 = fn.Node([2,0],2,2)
    node6 = fn.Node([2,1],2,2)
    quad = mg.Quadrilateral([node1,node2,node3,node4])
    #quad.getQuadMesh()
    #quad.plotQuadMesh()
    #quadtest = Quadrilateral([node1,node2,node4,node3],[0,1,3,2])
    #print(quadtest == quad)
    quadtest = mg.Quadrilateral([node2,node5,node6,node3])
    #quadtest.addNode(node1)
    #quadtest.addNode(node2)
    #quadtest.addNode(node5)
    #quadtest.addNode(node6)
    geo = mg.Geometry()
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
    fig = geo.plotMesh()
    [nodes, elems, mats, bdls] = geo.getMesh(fn.Node,mg.nodesQuad9,2)
    for node in nodes:
        pl.plot(node.getX()[0],node.getX()[1],'xb')
        
    elements = []
    for e in elems:
        elements.append(fe.StandardElement(e,[2,2],None,[i for i in range(9)],\
        None,None))

#test1()

class test_material(MAT.Material):
    def __init__(self, ID):
        self.ID = ID
    def getID(self):
        return self.ID

def test2():
    nodes = []
    nodes.append(fn.Node([0.0,-0.2],2,2))
    nodes.append(fn.Node([0.015,-0.2],2,2))
    nodes.append(fn.Node([0.0225,-0.2],2,2))
    nodes.append(fn.Node([0.0325,-0.2],2,2))
    nodes.append(fn.Node([0.2,-0.2],2,2))
    
    edges = [mg.Edge(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    geo = mg.Geometry()
    d = np.array([0.0,1.0])
    s = [0.1,0.064,0.015,0.0135,0.015,0.0135,0.015,0.064,0.1]
    for e in edges:
        geo.addPolygons(e.extendToQuad(d,s))
    
    polys = geo.getPolygons()
    for i in range(9):
        polys[i].setDivisionEdge13(10)
        
    for i in range(9,18):
        polys[i].setDivisionEdge13(2)
        
    for i in range(27,36):
        polys[i].setDivisionEdge13(5)
        
    for i in range(0,28,9):
        polys[i].setDivisionEdge24(4)
        
    for i in range(8,36,9):
        polys[i].setDivisionEdge24(4)
        
    for i in range(1,29,9):
        polys[i].setDivisionEdge24(2)
        
    for i in range(7,35,9):
        polys[i].setDivisionEdge24(4)
        
    mat1 = test_material(1)
    mat2 = test_material(2)
    mat3 = test_material(3)
    
    for i in range(1,8):
        polys[i].setMaterial(mat1)
        
    polys[20].setMaterial(mat2)
    polys[22].setMaterial(mat2)
    polys[24].setMaterial(mat2)
    
    for poly in polys:
        if poly.getMaterial() is None:
            poly.setMaterial(mat3)
        
    geo.mesh()
    
    [nodesx, elems, mats, bdls] = geo.getMesh(fn.Node,mg.nodesQuad9,2)
        
    #fig = geo.plot(poly_number = True, fill_mat = True)
        
    geo.plotMesh(col = 'b-',fill_mat = True)
    #for i,node in enumerate(nodesx):
    #    #pl.plot(node.getX()[0],node.getX()[1],'.b')
    #    if math.fabs(node.getX()[0] - 0.0)<1.0e-14:
    #        pl.text(node.getX()[0],node.getX()[1],str(i))
       
    for n in nodesx:
        if math.fabs(n.getX()[0]-0.0)<1.0e-14 or \
        math.fabs(n.getX()[1]+0.2)<1.0e-14 or \
        math.fabs(n.getX()[0]-0.2)<1.0e-14 or \
        math.fabs(n.getX()[1]-0.2)<1.0e-14:
            n.setConstraint(False, 0.0, 0)
            n.setConstraint(False, 0.0, 1)
            #pl.plot(n.getX()[0],n.getX()[1],'.r')
    
    elements = []
    for i,e in enumerate(elems):
        elements.append(fe.StandardElement(e,[2,2],None,[i for i in range(9)],\
        mats[i],None))
        if bdls[i] is not None:
            def loadfunc(t):
                return bdls[i]*math.sin(8.1e3*2*np.pi*t)
        else:
            loadfunc = None
        elements[i].setBodyLoad(loadfunc)
        
    return [nodesx, elements]
            
[nodes, elems] = test2()
    