# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:29:18 2018

@author: haiau
"""

import Mesh.FEMMesh as FM
import InOut.FEMOutput as FO
import Mesh.FEMNode as FN
import numpy as np
import pylab as pl

filen = '/media/haiau/Data/PyFEM_Results/result_100.dat'
Ndof = 2
tOrder = 2

res100,_ = FO.StandardFileOutput.readOutput(filen,val='x')

res = res100.tolist()
nodes = []
for x in res:
    nodes.append(FN.Node(x,Ndof,timeOrder=tOrder))

node = FM.findNodeNearX(nodes,np.array([0.015,0.0,0.0]))

#testres,t = FO.StandardFileOutput.readOutput(filen,list(range(100)),val='u')

fldrn = '/media/haiau/Data/PyFEM_Results/result_'

num_steps = [100,200,400,800,1600,3200]

def error_analysis(fldrn):
    
    resu = []
    rest = []
    
    for n in num_steps:
        filen = fldrn + str(n)+'.dat'
        print('reading data file: '+filen)
        testout,tout = FO.StandardFileOutput.readOutput(filen,timeStep='all',val='u')
        resu.append(testout)
        rest.append(tout)
    
    erru = [[]]
    
    for i in range(1,len(num_steps)):
        maxu = np.max([np.max(r[:,0]) for r in resu[i]])
        for j in range(len(rest[i-1])):
            uref = resu[i][j*2][:,0]
            ucur = resu[i-1][j][:,0]
            erru[-1].append(np.linalg.norm(uref-ucur)/maxu)
        erru.append([])
        
    averr = []
    for i in range(1,len(num_steps)):
        averr.append(np.average(erru[i-1]))
        
    aparm = np.polyfit(np.log10([1.0e-3/n for n in num_steps[0:-1]]),np.log10(averr),1)
        
    return resu,erru,rest,aparm,averr
    
resu_nl,erru_nl,rest,aparm_nl,averr_nl = error_analysis(fldrn)    

fldrn = '/media/haiau/Data/PyFEM_Results/result_linear_'

resu_lin,erru_lin,rest,aparm_lin,averr_lin = error_analysis(fldrn)

fldrn = '/media/haiau/Data/PyFEM_Results/result_linear_si_'
    
resu_lin_si,erru_lin_si,rest,aparm_lin_si,averr_lin_si = error_analysis(fldrn)
    
#fig = pl.Figure()
#
for i in range(1,len(num_steps)):
    pl.semilogy(rest[i-1],erru_nl[i-1],label='$\Delta_t=$'+str(1.0e-3/num_steps[i-1])+'$s$')

pl.legend(loc=4,prop={'size': 8})
pl.xlabel('$t$')
pl.ylabel('$e$')

fig = pl.Figure()

for i in range(1,len(num_steps)):
    pl.semilogy(rest[i-1],erru_lin[i-1],label='$\Delta_t=$'+str(1.0e-3/num_steps[i-1])+'$s$')

pl.legend(loc=4,prop={'size': 8})
pl.xlabel('$t$')
pl.ylabel('$e$')

fig = pl.Figure()

for i in range(1,len(num_steps)):
    pl.semilogy(rest[i-1],erru_lin_si[i-1],label='$\Delta_t=$'+str(1.0e-3/num_steps[i-1])+'$s$')

pl.legend(loc=4,prop={'size': 8})
pl.xlabel('$t$')
pl.ylabel('$e$')



fig = pl.figure()

resun = [[]]
for i in range(len(num_steps)):
    for j in range(len(rest[i])):
        resun[i].append(resu_nl[i][j][node[1],0])
    resun.append([])
        
resun_lin = [[]]
for i in range(len(num_steps)):
    for j in range(len(rest[i])):
        resun_lin[i].append(resu_lin[i][j][node[1],0])
    resun_lin.append([])        

resun_lin_si = [[]]
for i in range(len(num_steps)):
    for j in range(len(rest[i])):
        resun_lin_si[i].append(resu_lin_si[i][j][node[1],0])
    resun_lin_si.append([])           
        
err_loc = []
for i in range(len(num_steps)-1):
    errl = np.linalg.norm(np.array(\
    resun_lin[i]-np.array(\
    [resun_lin[i+1][j] for j in range(len(resun_lin[i]))])))/np.linalg.norm(\
    np.array(\
    [resun_lin[i+1][j] for j in range(len(resun_lin[i]))])) 
    err_loc.append(errl)
       
        
for i in range(len(num_steps)):
    pl.plot(rest[i],resun[i],label='$\Delta_t=$'+str(1.0e-3/num_steps[i-1])+'$s$')

pl.legend(prop={'size': 10})
pl.xlabel('$t$[$s$]')
pl.ylabel('$A$[$Vms^{-1}$]')

fig = pl.figure()
#
pl.loglog([1.0e-3/n for n in num_steps[0:-1]],averr_nl,'-x',label='Nonlinear')
pl.loglog([1.0e-3/n for n in num_steps[0:-1]],averr_lin,'-x',label='Linear with standard Gaussian Quadrature')
pl.loglog([1.0e-3/n for n in num_steps[0:-1]],averr_lin_si,'-x',label='Linear with singular Gaussian Quadrature')
pl.legend(loc=2,prop={'size':10})
pl.xlabel('$\Delta t$[$s$]')
pl.ylabel('e')
    
#pl.loglog([1.0e-3/n for n in num_steps[0:-1]],averr)

