# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:07:40 2017

@author: haiau
"""

import numpy as np
import itertools as it

class IntegrationData(object):
    """
    Integration Data
    Container stores information about the integration
    """
    pass

class GaussianQuadrature(IntegrationData):
    """
    Gaussian Quadrature
    This class stores information about the Gauss points
    This container has iterator that loops over Gauss points
    """
    def __init__(self, Ng, ndim, gen):
        """
        Initialize Gaussian Quadrature
        Input:
            Ng: number of Gauss Points, array of size ndim
            ndim: number of dimension
            gen: function that generate 1 dimensional Gaussian points
                this function must return a tuple (xg, wg) corresponds to
                xg: gauss points, array
                wg: weight numbers, array
                if gen is None, you have to generate self.xg and self.wg
                somewhere before the call to iterator
        """
        ng = np.array(Ng)
#        if ng.size < ndim:
#            raise Exception()
        if ndim < 1 and ndim > 3:
            raise UnsupportedGaussQuadrature
        self.Ng = ng
        self.Ndim = ndim
        self.Npoint = np.prod(Ng)
        self.xg = None
        self.wg = None
        if not gen is None and callable(gen):
            self.generatePoints(gen)
        self.iter_index = 0
        
    def __iter__(self):
        if self.xg is None or self.wg is None:
            raise GaussianDataNotGenerated
        self.iter_index = 0
        return self
        
    def __next__(self):
        if self.iter_index == self.Npoint:
            raise StopIteration
        idx = self.iter_index
        self.iter_index += 1
        if self.Ndim == 1:
            return self.xg[idx], self.wg[idx]
        try:
            return self.xg[:,idx], self.wg[:,idx]
        except IndexError:
            return self.xg[:,idx], self.wg[idx]
        
    def s_iter(self, op):
        """
        second iterator
        default behaviour is similar to default iterator
        This iterator can be override to have different behaviour, for example,
        in boundary element integration
        Notice: this generator yields one Gaussian point and two weights at one
        loop step.
        """
        for i in range(self.Ng):
            yield self.xg[i], self.wg[i], self.wg[i]
            
    def t_iter(self, op):
        """
        third iterator
        default behaviour is similar to default iterator
        This iterator can be override to have different behaviour, for example,
        in boundary element integration
        Notice: this generator yields two Gaussian points and two weights at 
        one loop step.
        """
        for i in range(self.Ng):
            yield self.xg[i], self.wg[i], self.xg[i], self.wg[i]
        
    def getNumberPoint(self):
        """
        Return total number of Gaussian Quadrature points
        """
        return self.Npoint
        
    def getDataAt(self, i):
        """
        Return i-th Gaussian data
        """
        if self.xg is None or self.wg is None:
            raise GaussianDataNotGenerated
        if self.Ndim == 1:
            return self.xg[i], self.wg[i]
        return self.xg[:,i], self.wg[:,i]
        
    def generatePoints(self, gen):
        if self.Ndim == 1:
            self.xg, self.wg = gen(np.int(self.Ng))
        if self.Ndim == 2:
            xg1, wg1 = gen(self.Ng[0])
            xg2, wg2 = gen(self.Ng[1])
            self.xg = np.array(list(it.product(xg1,xg2))).transpose()
            self.wg = np.array(list(it.product(wg1,wg2))).transpose()
        if self.Ndim == 3:
            xg1, wg1 = gen(self.Ng[0])
            xg2, wg2 = gen(self.Ng[1])
            xg3, wg3 = gen(self.Ng[2])
            self.xg = np.array(list(it.product(xg1,xg2,xg3))).transpose()
            self.wg = np.array(list(it.product(wg1,wg2,xg3))).transpose()
            
class GaussianQuadratureFile(GaussianQuadrature):
    def __init__(self, filename):
        self.file = open(filename,'r')
        header = self.file.readline()
        header = header.split()
        ng = int(header[1])
        ndim = 1
        GaussianQuadrature.__init__(self,ng,ndim,None)
        self.xg = np.zeros(ng,'float64')
        self.wg = np.zeros(ng,'float64')
        for i in range(ng):
            line = self.file.readline()
            dat = line.split()
            self.xg[i] = float(dat[0])
            self.wg[i] = float(dat[1])

class GaussianQuadratureOnEdge(GaussianQuadrature):
    def __init__(self, Ng, gen, edg):
        GaussianQuadrature.__init__(self,Ng,2,None)
        self.edg = edg
        if edg == 1:
            xg1, wg1 = gen(self.Ng[0])
            xg2 = np.ones(self.Ng[1],'float64')
            xg2 *= -1.0
            wg2 = np.ones(self.Ng[1],'float64')
        elif edg == 3:
            xg1, wg1 = gen(self.Ng[0])
            xg2 = np.ones(self.Ng[1],'float64')
            wg2 = np.ones(self.Ng[1],'float64')
        elif edg == 2:
            xg2, wg2 = gen(self.Ng[0])
            xg1 = np.ones(self.Ng[1],'float64')
            wg1 = np.ones(self.Ng[1],'float64')
        elif edg == 4:
            xg2, wg2 = gen(self.Ng[0])
            xg1 = np.ones(self.Ng[1],'float64')
            xg1 *= -1.0
            wg1 = np.ones(self.Ng[1],'float64')
        else:
            raise ValueError
        self.xg = np.array(list(it.product(xg1,xg2))).transpose()
        self.wg = np.array(list(it.product(wg1,wg2))).transpose()

def integrateGauss(func, xb, xe, xg, wg):
    Jacobi = (xe-xb)/2.0
    def x_(xi):
        return Jacobi*xi+xb
    ng = len(xg)
    res = 0
    for i in range(ng):
        res += func(x_(xg(i)))*wg(i)
    return res

        
def Gaussian1D(ng):
    xi = np.empty(ng)
    alpha = np.empty(ng)
    if ng == 1:
        xi   [ 0]  = 0.0
        alpha[ 0]  = 2.0
    #---------------------------------------------- 2 gauss points ---------
    elif ng == 2:
        xi   [ 0]  = -.5773502691896257
        xi   [ 1]  = 0.5773502691896257
        alpha[ 0]  = 1.0000000000000000
        alpha[ 1]  = 1.0000000000000000
 
    #---------------------------------------------- 3 gauss points ---------
    elif ng == 3:
        xi[0]  = -.7745966692414834
        xi[1]  = 0.0000000000000000
        xi[2]  = 0.7745966692414834
        alpha[ 0]  = 0.5555555555555556
        alpha[ 1]  = 0.8888888888888891
        alpha[ 2]  = 0.5555555555555556
	    
	    #---------------------------------------------- 4 gauss points ---------
    elif ng == 4:
        xi   [ 0]  = -.8611363115940526
        xi   [ 1]  = -.3399810435848563
        xi   [ 2]  = 0.3399810435848563
        xi   [ 3]  = 0.8611363115940526
        alpha[ 0]  = 0.3478548451374540
        alpha[ 1]  = 0.6521451548625461
        alpha[ 2]  = 0.6521451548625459
        alpha[ 3]  = 0.3478548451374539
	    
	    #---------------------------------------------- 5 gauss points ---------
    elif ng ==5:
        xi   [ 0]  = -.9061798459386640
        xi   [ 1]  = -.5384693101056830
        xi   [ 2]  = 0.0000000000000000
        xi   [ 3]  = 0.5384693101056830
        xi   [ 4]  = 0.9061798459386640
        alpha[ 0]  = 0.2369268850561889
        alpha[ 1]  = 0.4786286704993663
        alpha[ 2]  = 0.5688888888888879
        alpha[ 3]  = 0.4786286704993667
        alpha[ 4]  = 0.2369268850561890
	    
	    #---------------------------------------------- 6 gauss points ---------
    elif ng == 6:
        xi   [ 0]  = -.9324695142031521
        xi   [ 1]  = -.6612093864662646
        xi   [ 2]  = -.2386191860831969
        xi   [ 3]  = 0.2386191860831969
        xi   [ 4]  = 0.6612093864662646
        xi   [ 5]  = 0.9324695142031521
        alpha[ 0]  = 0.1713244923791714
        alpha[ 1]  = 0.3607615730481379
        alpha[ 2]  = 0.4679139345726927
        alpha[ 3]  = 0.4679139345726901
        alpha[ 4]  = 0.3607615730481377
        alpha[ 5]  = 0.1713244923791709
	    
    #---------------------------------------------- 7 gauss points ---------
    elif ng == 7:
        xi   [ 0]  = -.9491079123427584
        xi   [ 1]  = -.7415311855993945
        xi   [ 2]  = -.4058451513773971
        xi   [ 3]  = 0.000000000000000
        xi   [ 4]  = 0.4058451513773971
        xi   [ 5]  = 0.7415311855993945
        xi   [ 6]  = 0.9491079123427584
        alpha[ 0]  = 0.1294849661688690
        alpha[ 1]  = 0.2797053914892758
        alpha[ 2]  = 0.3818300505051191
        alpha[ 3]  = 0.4179591836734642
        alpha[ 4]  = 0.3818300505051198
        alpha[ 5]  = 0.2797053914892777
        alpha[ 6]  = 0.1294849661688699
	    
    #---------------------------------------------- 8 gauss points ---------
    elif ng == 8:
        xi   [ 0]  = -.9602898564975368
        xi   [ 1]  = -.7966664774136262
        xi   [ 2]  = -.5255324099163290
        xi   [ 3]  = -.1834346424956498
        xi   [ 4]  = 0.1834346424956498
        xi   [ 5]  = 0.5255324099163290
        xi   [ 6]  = 0.7966664774136262
        xi   [ 7]  = 0.9602898564975368
        alpha[ 0]  = 0.1012285362903747
        alpha[ 1]  = 0.2223810344533735
        alpha[ 2]  = 0.3137066458778813
        alpha[ 3]  = 0.3626837833783669
        alpha[ 4]  = 0.3626837833783578
        alpha[ 5]  = 0.3137066458778852
        alpha[ 6]  = 0.2223810344533744
        alpha[ 7]  = 0.1012285362903746
	   
    #---------------------------------------------- 9 gauss points ---------
    elif ng == 9:
        xi   [ 0]  = -.9681602395076256
        xi   [ 1]  = -.8360311073266362
        xi   [ 2]  = -.6133714327005904
        xi   [ 3]  = -.3242534234038089
        xi   [ 4]  = 0.0000000000000000
        xi   [ 5]  = 0.3242534234038089
        xi   [ 6]  = 0.6133714327005904
        xi   [ 7]  = 0.8360311073266362
        xi   [ 8]  = 0.9681602395076256
        alpha[ 0]  = 0.0812743883615748
        alpha[ 1]  = 0.1806481606948518
        alpha[ 2]  = 0.2606106964029281
        alpha[ 3]  = 0.3123470770399878
        alpha[ 4]  = 0.3302393550012641
        alpha[ 5]  = 0.3123470770400267
        alpha[ 6]  = 0.2606106964029345
        alpha[ 7]  = 0.1806481606948535
        alpha[ 8]  = 0.0812743883615771
	   
    #----------------------------------------------10 gauss points ---------
    elif ng == 10:
        xi   [ 0]  = -.9739065285171716
        xi   [ 1]  = -.8650633666889844
        xi   [ 2]  = -.6794095682990247
        xi   [ 3]  = -.4333953941292470
        xi   [ 4]  = -.1488743389816312
        xi   [ 5]  = 0.1488743389816312
        xi   [ 6]  = 0.4333953941292470
        xi   [ 7]  = 0.6794095682990247
        xi   [ 8]  = 0.8650633666889844
        xi   [ 9]  = 0.9739065285171716
        alpha[ 0]  = 0.0666713443086864
        alpha[ 1]  = 0.1494513491505734
        alpha[ 2]  = 0.2190863625159556
        alpha[ 3]  = 0.2692667193100258
        alpha[ 4]  = 0.2955242247147621
        alpha[ 5]  = 0.2955242247147297
        alpha[ 6]  = 0.2692667193099739
        alpha[ 7]  = 0.2190863625159674
        alpha[ 8]  = 0.1494513491506039
        alpha[ 9]  = 0.0666713443086882
    #----------------------------------------------11 gauss points ---------
    elif ng == 11:
        xi   [ 0]  = -.9782286581460600
        xi   [ 1]  = -.8870625997680912
        xi   [ 2]  = -.7301520055740505
        xi   [ 3]  = -.5190961292068119
        xi   [ 4]  = -.2695431559523450
        xi   [ 5]  = 0.0000000000000000
        xi   [ 6]  = 0.2695431559523450
        xi   [ 7]  = 0.5190961292068119
        xi   [ 8]  = 0.7301520055740505
        xi   [ 9]  = 0.8870625997680912
        xi   [10]  = 0.9782286581460600
        alpha[ 0]  = 0.0556685671161702
        alpha[ 1]  = 0.1255803694648913
        alpha[ 2]  = 0.1862902109277069
        alpha[ 3]  = 0.2331937645920298
        alpha[ 4]  = 0.2628045445102398
        alpha[ 5]  = 0.2729250867778585
        alpha[ 6]  = 0.2628045445102415
        alpha[ 7]  = 0.2331937645920817
        alpha[ 8]  = 0.1862902109277551
        alpha[ 9]  = 0.1255803694649200
        alpha[10]  = 0.0556685671161749

    #----------------------------------------------12 gauss points ---------
    elif ng == 12:
        xi   [ 0]  = -.9815606342467348
        xi   [ 1]  = -.9041172563704474
        xi   [ 2]  = -.7699026741943191
        xi   [ 3]  = -.5873179542866147
        xi   [ 4]  = -.3678314989981804
        xi   [ 5]  = -.1252334085114689
        xi   [ 6]  = 0.1252334085114689
        xi   [ 7]  = 0.3678314989981804
        xi   [ 8]  = 0.5873179542866147
        xi   [ 9]  = 0.7699026741943191
        xi   [10]  = 0.9041172563704474
        xi   [11]  = 0.9815606342467348
        alpha[ 0]  = 0.0471753363864993
        alpha[ 1]  = 0.1069393259953444
        alpha[ 2]  = 0.1600783285432111
        alpha[ 3]  = 0.2031674267231830
        alpha[ 4]  = 0.2334925365384291
        alpha[ 5]  = 0.2491470458133551
        alpha[ 6]  = 0.2491470458132625
        alpha[ 7]  = 0.2334925365384953
        alpha[ 8]  = 0.2031674267229132
        alpha[ 9]  = 0.1600783285433607
        alpha[10]  = 0.1069393259953818
        alpha[11]  = 0.0471753363865106
        
    #----------------------------------------------13 gauss points ---------
    elif ng == 13:
        xi   [ 0]  = -.9841830547185922
        xi   [ 1]  = -.9175983992229760
        xi   [ 2]  = -.8015780907333058
        xi   [ 3]  = -.6423493394403423
        xi   [ 4]  = -.4484927510364468
        xi   [ 5]  = -.2304583159551348
        xi   [ 6]  = 0.0000000000000000
        xi   [ 7]  = 0.2304583159551348
        xi   [ 8]  = 0.4484927510364468
        xi   [ 9]  = 0.6423493394403423
        xi   [10]  = 0.8015780907333058
        xi   [11]  = 0.9175983992229760
        xi   [12]  = 0.9841830547185922
        alpha[ 0]  = 0.0404840047652838
        alpha[ 1]  = 0.0921214998377965
        alpha[ 2]  = 0.1388735102193546
        alpha[ 3]  = 0.1781459807624038
        alpha[ 4]  = 0.2078160475371783
        alpha[ 5]  = 0.2262831802624006
        alpha[ 6]  = 0.2325515532300595
        alpha[ 7]  = 0.2262831802628873
        alpha[ 8]  = 0.2078160475371335
        alpha[ 9]  = 0.1781459807617468
        alpha[10]  = 0.1388735102195631
        alpha[11]  = 0.0921214998377338
        alpha[12]  = 0.0404840047653005

    #----------------------------------------------14 gauss points ---------
    elif ng == 14:
        xi   [ 0]  = -.9862838086968272
        xi   [ 1]  = -.9284348836635448
        xi   [ 2]  = -.8272013150697833
        xi   [ 3]  = -.6872929048116804
        xi   [ 4]  = -.5152486363581545
        xi   [ 5]  = -.3191123689278897
        xi   [ 6]  = -.1080549487073437
        xi   [ 7]  = 0.1080549487073437
        xi   [ 8]  = 0.3191123689278897
        xi   [ 9]  = 0.5152486363581545
        xi   [10]  = 0.6872929048116804
        xi   [11]  = 0.8272013150697833
        xi   [12]  = 0.9284348836635448
        xi   [13]  = 0.9862838086968272
        alpha[ 0]  = 0.0351194603317560
        alpha[ 1]  = 0.0801580871597245
        alpha[ 2]  = 0.1215185706870852
        alpha[ 3]  = 0.1572031671580320
        alpha[ 4]  = 0.1855383974768793
        alpha[ 5]  = 0.2051984637207979
        alpha[ 6]  = 0.2152638534625784
        alpha[ 7]  = 0.2152638534628823
        alpha[ 8]  = 0.2051984637213401
        alpha[ 9]  = 0.1855383974781713
        alpha[10]  = 0.1572031671582824
        alpha[11]  = 0.1215185706880219
        alpha[12]  = 0.0801580871598046
        alpha[13]  = 0.0351194603317396

    #----------------------------------------------15 gauss points ---------
    elif ng == 15:
	    xi   [ 0]  = -.9879925180205082
	    xi   [ 1]  = -.9372733924006658
	    xi   [ 2]  = -.8482065834104466
	    xi   [ 3]  = -.7244177313601691
	    xi   [ 4]  = -.5709721726085377
	    xi   [ 5]  = -.3941513470775635
	    xi   [ 6]  = -.2011940939974345
	    xi   [ 7]  = 0.0000000000000000
	    xi   [ 8]  = 0.2011940939974345
	    xi   [ 9]  = 0.3941513470775635
	    xi   [10]  = 0.5709721726085377
	    xi   [11]  = 0.7244177313601691
	    xi   [12]  = 0.8482065834104466
	    xi   [13]  = 0.9372733924006658
	    xi   [14]  = 0.9879925180205082
	    alpha[ 0]  = 0.0307532419962281
	    alpha[ 1]  = 0.0703660474878736
	    alpha[ 2]  = 0.1071592204670246
	    alpha[ 3]  = 0.1395706779269623
	    alpha[ 4]  = 0.1662692058175702
	    alpha[ 5]  = 0.1861610000180186
	    alpha[ 6]  = 0.1984314853286876
	    alpha[ 7]  = 0.2025782419235544
	    alpha[ 8]  = 0.1984314853276636
	    alpha[ 9]  = 0.1861610000173629
	    alpha[10]  = 0.1662692058148672
	    alpha[11]  = 0.1395706779251018
	    alpha[12]  = 0.1071592204664661
	    alpha[13]  = 0.0703660474879402
	    alpha[14]  = 0.0307532419961173

    #----------------------------------------------16 gauss points ---------
    elif ng == 16:
	    xi   [ 0]  = -.9894009349917192
	    xi   [ 1]  = -.9445750230731024
	    xi   [ 2]  = -.8656312023879067
	    xi   [ 3]  = -.7554044083549896
	    xi   [ 4]  = -.6178762444026428
	    xi   [ 5]  = -.4580167776572275
	    xi   [ 6]  = -.2816035507792589
	    xi   [ 7]  = -.0950125098376374
	    xi   [ 8]  = 0.0950125098376374
	    xi   [ 9]  = 0.2816035507792589
	    xi   [10]  = 0.4580167776572275
	    xi   [11]  = 0.6178762444026428
	    xi   [12]  = 0.7554044083549896
	    xi   [13]  = 0.8656312023879067
	    xi   [14]  = 0.9445750230731024
	    xi   [15]  = 0.9894009349917192
	    alpha[ 0]  = 0.0271524594116876
	    alpha[ 1]  = 0.0622535239382955
	    alpha[ 2]  = 0.0951585116800162
	    alpha[ 3]  = 0.1246289712557735
	    alpha[ 4]  = 0.1495959888171822
	    alpha[ 5]  = 0.1691565193942059
	    alpha[ 6]  = 0.1826034150474125
	    alpha[ 7]  = 0.1894506104520706
	    alpha[ 8]  = 0.1894506104555478
	    alpha[ 9]  = 0.1826034150405641
	    alpha[10]  = 0.1691565193904098
	    alpha[11]  = 0.1495959888122324
	    alpha[12]  = 0.1246289712567469
	    alpha[13]  = 0.0951585116826968
	    alpha[14]  = 0.0622535239380804
	    alpha[15]  = 0.0271524594119934

    #----------------------------------------------17 gauss points ---------
    elif ng == 17:
	    xi   [ 0]  = -.9905754753146424
	    xi   [ 1]  = -.9506755217683084
	    xi   [ 2]  = -.8802391537273105
	    xi   [ 3]  = -.7815140038966982
	    xi   [ 4]  = -.6576711592167042
	    xi   [ 5]  = -.5126905370864762
	    xi   [ 6]  = -.3512317634538764
	    xi   [ 7]  = -.1784841814958479
	    xi   [ 8]  = 0.0000000000000000
	    xi   [ 9]  = 0.1784841814958479
	    xi   [10]  = 0.3512317634538764
	    xi   [11]  = 0.5126905370864762
	    xi   [12]  = 0.6576711592167042
	    xi   [13]  = 0.7815140038966982
	    xi   [14]  = 0.8802391537273105
	    xi   [15]  = 0.9506755217683084
	    xi   [16]  = 0.9905754753146424
	    alpha[ 0]  = 0.0241483028690637
	    alpha[ 1]  = 0.0554595293748953
	    alpha[ 2]  = 0.0850361483133380
	    alpha[ 3]  = 0.1118838471995337
	    alpha[ 4]  = 0.1351363684638741
	    alpha[ 5]  = 0.1540457610746122
	    alpha[ 6]  = 0.1680041021542135
	    alpha[ 7]  = 0.1765627053813453
	    alpha[ 8]  = 0.1794464703549354
	    alpha[ 9]  = 0.1765627053687074
	    alpha[10]  = 0.1680041021477990
	    alpha[11]  = 0.1540457610639987
	    alpha[12]  = 0.1351363684820280
	    alpha[13]  = 0.1118838471934820
	    alpha[14]  = 0.0850361483134939
	    alpha[15]  = 0.0554595293744238
	    alpha[16]  = 0.0241483028685680

    #----------------------------------------------18 gauss points ---------
    elif ng == 18:
	    xi   [ 0]  = -.9915651684207047
	    xi   [ 1]  = -.9558239495719209
	    xi   [ 2]  = -.8926024664970845
	    xi   [ 3]  = -.8037049589727451
	    xi   [ 4]  = -.6916870430603012
	    xi   [ 5]  = -.5597708310739521
	    xi   [ 6]  = -.4117511614628424
	    xi   [ 7]  = -.2518862256915055
	    xi   [ 8]  = -.0847750130417353
	    xi   [ 9]  = 0.0847750130417353
	    xi   [10]  = 0.2518862256915055
	    xi   [11]  = 0.4117511614628424
	    xi   [12]  = 0.5597708310739521
	    xi   [13]  = 0.6916870430603012
	    xi   [14]  = 0.8037049589727451
	    xi   [15]  = 0.8926024664970845
	    xi   [16]  = 0.9558239495719209
	    xi   [17]  = 0.9915651684207047
	    alpha[ 0]  = 0.0216160135261377
	    alpha[ 1]  = 0.0497145488938536
	    alpha[ 2]  = 0.0764257302572466
	    alpha[ 3]  = 0.1009420440958779
	    alpha[ 4]  = 0.1225552067059565
	    alpha[ 5]  = 0.1406429146838828
	    alpha[ 6]  = 0.1546846751121419
	    alpha[ 7]  = 0.1642764837630347
	    alpha[ 8]  = 0.1691423829486400
	    alpha[ 9]  = 0.1691423829263725
	    alpha[10]  = 0.1642764837871328
	    alpha[11]  = 0.1546846750990626
	    alpha[12]  = 0.1406429146606422
	    alpha[13]  = 0.1225552066966724
	    alpha[14]  = 0.1009420441000757
	    alpha[15]  = 0.0764257302475851
	    alpha[16]  = 0.0497145488967877
	    alpha[17]  = 0.0216160135259334
    else:
        return np.polynomial.legendre.leggauss(ng)
    return xi, alpha
        
class UnsupportedGaussQuadrature(Exception):
    pass

class GaussianDataNotGenerated(Exception):
    pass