# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:38:48 2018

@author: haiau
"""

import Mesh.FEMMesh as FM
import InOut.FEMOutput as FO
import Mesh.FEMNode as FN
import numpy as np
import pylab as pl
import matplotlib.ticker as ticker

fldrn = '/media/haiau/Data/PyFEM_Results/result_linear_ndiv_'

Ndof = 2
tOrder = 2

ndiv = [1,2,4,8]
res = []

for n in ndiv:
    filen = fldrn + str(n)+'.dat'
    print('reading data file: '+filen)
    res100,_ = FO.StandardFileOutput.readOutput(filen,val='x')
    resx = res100.tolist()
    nodes = []
    for x in resx:
        nodes.append(FN.Node(x,Ndof,timeOrder=tOrder))
    _,inode = FM.findNodeNearX(nodes,np.array([0.015,-0.1,0.0]))
    testout,tout = FO.StandardFileOutput.readOutput(filen,timeStep='all',node=inode,val='u')
    rest = [t[0][0] for t in testout]
    res.append(np.array(rest))
    
err = []    
    
for i,n in enumerate(ndiv[0:-1]):
    err.append(np.linalg.norm(res[i+1]-res[i])/np.linalg.norm(res[i+1]))
    
#pl.plot(ndiv[0:-1],err)    
pl.plot(ndiv,np.array(res)[:,200],'-x')
pl.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2e'))  
#testres,t = FO.StandardFileOutput.readOutput(filen,list(range(100)),val='u')


