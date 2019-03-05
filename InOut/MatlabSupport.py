# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:59:50 2017

@author: haiau
"""

import matlab
import Mesh.FEMMesh as FM

def plot_field(eng, mesh, val='u'):
    Ne = mesh.Ne
    it = FM.get_connections(mesh)
    mat = []
    for e in mesh.getElements():
        mat.append(e.getMaterial().getID()+1)
        
    xyz = []
    field = []
    for n in mesh.getNodes():
        xyz.append(n.getX().tolist())
    if val == 'u':
        for n in mesh.getNodes():
            field.append(n.getU().tolist())
    elif val == 'v':
        for n in mesh.getNodes():
            field.append(n.getV().tolist())
    elif val == 'a':
        for n in mesh.getNodes():
            field.append(n.getU().tolist())
    else:
        return
        
    #eng = matlab.engine.start_matlab()
    IT = matlab.int64(it)
    MAT = matlab.int64(mat)
    XYZ = matlab.double(xyz)
    Field = matlab.double(field)
    eng.plot_field(IT,Field,XYZ,Ne,MAT,0)
    return eng
    