# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:42:21 2017

@author: haiau
"""

import numpy as np
import itertools as itl

__all__ = ['injectArray','zeros','array']

class injectArray(object):
    """
    Array of different elements of other array, only support maximum 2-d array
    """
    def __init__(self, size, dtype = 'float64', data = None):
        """
        Initialize injection array
        Input:
            size: either size of 1d array or tuple of sizes of dimensions
            dtype: data type of each element, similar to numpy array datatype
        """
        assert isinstance(size,(list,tuple,int)),'Wrong array size'
        assert dtype is not None, 'unsupported data type'
        self.dtype = dtype
        self.size = size
        try:
            self.ndim = len(size)
            self.length = np.prod(size)
        except TypeError:
            self.length = size
            self.ndim = 1
            
        if data is not None:
            self.data = data
            return
            
        self.data = np.empty(size,dtype=np.ndarray)            
        
        a = np.array([0.0]*self.length,dtype)
        it =np.nditer(self.data,flags=['refs_ok','c_index','multi_index'],\
        op_flags=['readwrite'])
        while not it.finished:
            idx = it.index            
            try:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                self.data[id1,id2] = a[idx:idx+1]
            except IndexError:
                self.data[idx] = a[idx:idx+1]
            it.iternext()
            
    def __iter__(self):
        self.it = np.nditer(self.data,\
        flags=['refs_ok','c_index','multi_index'],op_flags=['readwrite'])
        return self
        
    def __next__(self):
        if self.it.finished:
            del self.it
            raise StopIteration
        try:
            id1 = self.it.multi_index[0]
            id2 = self.it.multi_index[1]
            self.it.iternext()
            return self.data[id1,id2][0]
        except:
            idx = self.it.index
            return self.data[idx][0]
            
    def __str__(self):
        """
        print array
        """
        s = '['
        try:
            for i in range(self.size[0]):
                
                try:
                    s += '['
                    for j in range(self.size[1]):
                        s+= str(self.data[i,j][0])
                        s+= ', '
                    s = s[:-2]
                    s+= '], '
                    #s = s[:-2]
                except IndexError:
                    s = s[:-1]
                    s+= str(self.data[i][0])
                    s+= ', '
            s = s[:-2]
        except TypeError:
            for i in range(self.size):
                s+= str(self.data[i][0])
                s+= ' ,'
            s = s[:-2]
        s+= ']'
        return s
        
    def __getitem__(self, k):
        return self.data[k][0]
        
    def __setitem__(self, k, val):
        self.data[k][0] = val
        
    def __iadd__(self, other):   
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    self.data[i] += other[i]
            except (TypeError, IndexError):
                for i in a:
                    self.data[i] += other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    self.data[id1,id2] += other[id1,id2]
                except (TypeError, IndexError):
                    self.data[id1,id2] += other
                it.iternext()
        return self
        
    def __add__(self, other):
        res = np.empty(self.size, self.dtype)
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    res[i] = self.data[i] + other[i]
            except (TypeError, IndexError):
                for i in a:
                    res[i] = self.data[i] + other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    res[id1,id2] = self.data[id1,id2] + other[id1,id2]
                except (TypeError, IndexError):
                    res[id1,id2] = self.data[id1,id2] + other
                it.iternext()
        return res
        
    def __isub__(self, other):        
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    self.data[i] -= other[i]
            except (TypeError, IndexError):
                for i in a:
                    self.data[i] -= other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    self.data[id1,id2] -= other[id1,id2]
                except (TypeError, IndexError):
                    self.data[id1,id2] -= other
                it.iternext()
        return self
        
    def __sub__(self, other):
        res = np.empty(self.size, self.dtype)
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    res[i] = self.data[i] - other[i]
            except (TypeError, IndexError):
                for i in a:
                    res[i] = self.data[i] - other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    res[id1,id2] = self.data[id1,id2] - other[id1,id2]
                except (TypeError, IndexError):
                    res[id1,id2] = self.data[id1,id2] - other
                it.iternext()
        return res
        
    def __imul__(self, other):       
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    self.data[i] *= other[i]
            except (TypeError, IndexError):
                for i in a:
                    self.data[i] *= other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    self.data[id1,id2] *= other[id1,id2]
                except (TypeError, IndexError):
                    self.data[id1,id2] *= other
                it.iternext()
        return self
        
    def __mul__(self, other):
        res = np.empty(self.size, self.dtype)
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    res[i] = self.data[i] * other[i]
            except (TypeError, IndexError):
                for i in a:
                    res[i] = self.data[i] * other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    res[id1,id2] = self.data[id1,id2] * other[id1,id2]
                except (TypeError, IndexError):
                    res[id1,id2] = self.data[id1,id2] * other
                it.iternext()
        return res
        
    def __idiv__(self, other):        
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    self.data[i] /= other[i]
            except (TypeError, IndexError):
                for i in a:
                    self.data[i] /= other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    self.data[id1,id2] /= other[id1,id2]
                except (TypeError, IndexError):
                    self.data[id1,id2] /= other
                it.iternext()
        return self
        
    def __div__(self, other):
        res = np.empty(self.size, self.dtype)
        if self.ndim == 1:
            a = range(self.size)
            try:
                for i in a:
                    res[i] = self.data[i] / other[i]
            except (TypeError, IndexError):
                for i in a:
                    res[i] = self.data[i] / other
        elif self.ndim == 2:
            it =np.nditer(self.data,flags=['refs_ok','multi_index'],\
            op_flags=['readwrite'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                try:
                    res[id1,id2] = self.data[id1,id2] / other[id1,id2]
                except (TypeError, IndexError):
                    res[id1,id2] = self.data[id1,id2] / other
                it.iternext()
        return res
        
    def __itruediv__(self, other):
        return self.__idiv__(other)
        
    def __truediv__(self, other):
        return self.__div__(other)
        
    def connect(self, k, A, ka):
        if isinstance(ka,(list,np.ndarray,tuple)):
            A[ka[0],ka[1]] = self.data[k][0]
            self.data[k] = A[ka[0]:ka[0]+1,ka[1]:ka[1]+1]
        else:
            A[ka] = self.data[k][0]
            self.data[k] = A[ka:ka+1]
            
    def tolist(self):
        if self.ndim == 1:
            return [self.data[i][0] for i in range(self.size)]
        elif self.ndim == 2:
            res = np.empty(self.size,self.dtype)
            it =np.nditer(self.data,flags=['refs_ok','multi_index'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                res[id1,id2] = self.data[id1,id2][0]
                it.iternext()
            return res.tolist()
        return []
        
    def toNumpy(self):
        if self.ndim == 1:
            return np.array([self.data[i][0] for i in range(self.size)])
        elif self.ndim == 2:
            res = np.empty(self.size,self.dtype)
            it =np.nditer(self.data,flags=['refs_ok','multi_index'])
            while not it.finished:
                id1 = it.multi_index[0]
                id2 = it.multi_index[1]
                res[id1,id2] = self.data[id1,id2][0]
                it.iternext()
            return res
        return None
        
def zeros(size, dtype = 'float64'):
    """
    Create a injection array with all member set to zero
    This is equivalent to injectArray(size,dtype)
    """
    return injectArray(size,dtype)
    
def array(data, dtype = None):
    """
    Create an injection array from an array like data
    """
        
    if dtype is None:
        try:
            dtype = data.dtype
        except AttributeError:
            try:
                for a in itl.chain.from_iterable(data):
                    if isinstance(a, float):
                        dtype = 'float64'
                        break
                    elif isinstance(a, int):
                        dtype = 'int32'
            except TypeError:
                for a in data:
                    if isinstance(a, float):
                        dtype = 'float64'
                        break
                    elif isinstance(a, int):
                        dtype = 'int32'

    try:
        size = data.shape
        try:
            size[1]
        except IndexError:
            size = size[0]
            
    except AttributeError:
        try:
            size = data.size
        except AttributeError:
            size1 = len(data)
            try:
                size2 = len(data[1])
                size = (size1,size2)
            except TypeError:
                size = len(data)
                    
    res = injectArray(size,dtype)
                    
    it = np.nditer(res.data,flags=['c_index','multi_index','refs_ok'],\
    op_flags=['readwrite'])
    while not it.finished:
        try:
            id1 = it.multi_index[0]
            id2 = it.multi_index[1]
            res.data[id1][id2][0] = data[id1][id2]
        except IndexError:
            idx = it.index
            res.data[idx][0] = data[idx]
        it.iternext()
            
    return res
    
            
if __name__ == '__main__':
    test = injectArray((2,2))       
    print(test[0,0])
    testCon = np.array([1,2,3,4,5,6],dtype = 'float64')
    test.connect((0,0),testCon,2)
    print('test[0,0] = ',test[0,0])
    test[0,0] = 4
    test += np.array([[0.5,0.5],[0.6,0.6]])
    test += 0.1
    print('test += operator', test)
    test *= 2.0
    test /= 0.5
    testlist = test.tolist()
    print('test tolist ',testlist)
    for e in test:
        print(e)
        
    testarr1 = array([1,2,3,4])
    print(testarr1)
    testarr2 = array([[1,2,3],[4.0,5,6]])
    print(testarr2)
    testarr3 = array(np.array([1.0,2.0,3.0]))
    print(testarr3)
    testarr4 = np.array(array([4,5,6.0]))
    print(testarr4)
    print(testarr3*2)
    print(testarr2/2)
    testarr5 = np.array([[4.0],[5.0],[6.0]])
    print(testarr5)
    testarr7 = array(np.array([[1,2,3],[4,5,6]]))
    print(testarr7)
    testarr4 += testarr3
    print(testarr4)
    testarr2[1,1] += 5
    print(testarr2)
    #testarr8 = zeros((3,3),dtype = 'float64')
    #testarr6 = np.outer(testarr3,testarr5,testarr8)
    #print(array(testarr6))
