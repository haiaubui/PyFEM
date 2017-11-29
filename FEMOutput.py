# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:52:32 2017

@author: haiau
"""

import sys
import numpy as np

class FEMOutput(object):
    """
    Output data
    """
    def __init__(self):
        """
        Initialize FEMOutput object
        """
        pass
            
    def outputData(self, data):
        """
        write data to output file
        """
        pass
    
    def finishOutput(self):
        """
        Finish output data
        Close all files or return all objects
        """
    
class FileOutput(FEMOutput):
    """
    Output to file
    """
    def __init__(self, outfile):
        """
        Initialize FileOutput object and open file
        Input:
            outfile: output filename
        """
        self.outfile = outfile
        try:
            self.file = open(self.outfile, "w")
        except IOError:
            sys.exit('Cannot open file '+self.outfile)
    
    def finishOutput(self):
        """
        Finish output data
        Close output file
        """
        self.file.close()
            
class StandardFileOutput(FileOutput):
    """
    This object class gives the output in simple standard format.
    The format of the output file is following:
    Each time step begin with
    NNOD data.getNnod() TIME data.getTime() ORDER data.getTimeOrder()
    and follow by NNOD rows, each row is data of one node with format:
    Ndof  X[0] X[1] X[2] U[0]...U[Ndof-1] V[0] ... V[Ndof-1] A[0] ... A[Ndof-1]
    if data.getTimeOrder() == 0, there is no V and A
                              1,             A                         
    """
    def writeX(self, node):
        self.file.write(str(node.getX()[0])+' ')
        if node.getNdim() > 1:
            self.file.write(str(node.getX()[1])+' ')
            if node.getNdim() == 3:
                self.file.write(str(node.getX()[2])+' ')
            else:
                self.file.write('0.0 ') 
        else:
            self.file.write('0.0 0.0 ')
            
    def writeU(self, node):
        for i in range(node.getNdof()):
            self.file.write(str(node.getU()[i])+' ')
            
    def writeV(self, node):
        for i in range(node.getNdof()):
            self.file.write(str(node.getV()[i])+' ')
    
    def writeA(self, node):
        for i in range(node.getNdof()):
            self.file.write(str(node.getA()[i])+' ')
            
    def writeHeader(self, data):
        self.file.write('NNOD '+str(data.getMesh().getNnod()))
        if data.getTimeOrder() > 0:
            self.file.write(' TIME '+str(data.getTime())+' ORDER '+\
            str(data.getTimeOrder())+'\n')
        else:
            self.file.write(' TIME 0.0 ORDER 0\n')
    
    def outputData(self, data):
        """
        write data to output file
        """
        self.writeHeader(data)
        nodes = data.getMesh().getNodes()
        ndof = nodes[0].getNdof()
        for i in range(data.getMesh().getNnod()):
            node = nodes[i]
            assert node.getNdof() == ndof,'Incompatible DOFs between nodes'
            self.file.write(str(node.getNdof())+' ')
            self.writeX(node)
            self.writeU(node)
            if data.getTimeOrder() > 0:
                self.writeV(node)
            if data.getTimeOrder() == 2:
                self.writeA(node)
            self.file.write('\n')
            
    def readHeader(self, file):
        hdr = file.readline()
        ex = Exception('The file is corrupted or false formatted!')
        if hdr == '':
            raise EOFError
        hdr = hdr.split()
        if hdr[0] != 'NNOD':
            raise ex
        try:
            Nnod = int(hdr[1])
        except:
            raise ex            
        if hdr[2] != 'TIME':
            raise ex
        try:
            t = int(hdr[3])
        except:
            raise ex
        if hdr[4] != 'ORDER':
            raise ex
        try:
            torder = int(hdr[5])
        except:
            raise ex
            
        return Nnod, t, torder
            
    def __readNodes(file, node, val, Nnod):
        res = []
        allnode = True and isinstance(node, str)
        
        for i in range(Nnod):            
            line = file.readline()
            if line is None:
                raise EOFError
            if not allnode and node != i:
                continue
            line = line.split()
            Ndof = int(line[0])
            try:
                if val == 'x':
                    X_ = list(float(x) for x in line[1:4])
                    res.append(X_)
                elif val == 'u':
                    u = list(float(x) for x in line[4:4+Ndof])
                    res.append(u)
                elif val == 'v':
                    v = list(float(x) for x in line[4+Ndof:4+Ndof*2])
                    res.append(v)
                elif val == 'a':
                    a = list(float(x) for x in line[4+Ndof*2:4+Ndof*3])
                    res.append(a)
                else:
                    raise Exception('Cannot read '+val+' in this file')
            except IndexError:
                raise Exception('Cannot read '+val+' in this file')
        return np.array(res)       
            
    def readOutput(file, timeStep = 0, node = 'all', val = 'u'):
        """
        This function read the ouput file that was produced by 
        this output class
        Input:
            file: file instance of output file
            timeStep:
                scalar: read the specific time step
                list, tuple: read the time steps in list or tuple
                'all': read all time steps
            node:
                scalar: read at specific node
                'all': read all nodes
            val:
                'u': read displacement u
                'v': read velocity v
                'a': read acceleration a
                'x': read coordinates of nodes
        Return:
            specific time step: numpy array
            other cases: list of numpy array
        """
        Nnod, ti, timeOrder = StandardFileOutput.readheader(file)
        assert not(isinstance(timeStep,str) and timeStep != 'all'),\
        'unknown option '+ str(timeStep)
            
        alltime = True and timeStep == 'all'
        res = []    
        if not alltime:
            if isinstance(timeStep,(list,tuple)):
                for i in timeStep:
                    res.append(StandardFileOutput.readOutput(file,i,node,val))
                return res
            else:        
                gotoLine(file, (Nnod+1)*timeStep)
                StandardFileOutput.readheader(file) 
                return StandardFileOutput.__readNodes(file,node,val,Nnod)
        try:
            file.seek(0)
            while 1:
               StandardFileOutput.readheader(file) 
               res.append(StandardFileOutput.__readNodes(file,node,val,Nnod))
        except EOFError:
            return res
            

def gotoLine(file, lineno):
    """
    This function goto the specific line in an opened file
    Input:
        file: file object that is opened
        lineno: line number
    Raise:
        EOFError if file ended before line numer is reached
    """
    file.seek(0)
    for i in range(lineno):
        line = file.readline()
        if line == '':
            raise EOFError
 
class NoXStardardFileOutput(StandardFileOutput):
    """
    This object class gives the output in simple standard format without
    coordinates of nodes.
    The format of the output file is following:
    Each time step begin with
    NNOD data.getNnod() TIME data.getTime() ORDER data.getTimeOrder()
    and follow by NNOD rows, each row is data of one node with format:
    Ndof  U[0]...U[Ndof-1] V[0] ... V[Ndof-1] A[0] ... A[Ndof-1]
    if data.getTimeOrder() == 0, there is no V and A
                              1,             A                         
    """
    def outputData(self, data):
        """
        write data to output file
        """
        self.writeHeader(data)
        nodes = data.getMesh().getNodes()
        for i in range(data.getMesh().getNnod()):
            node = nodes[i]
            self.file.write(str(node.getNdof())+' ')
            self.writeU(node)
            if data.getTimeOrder() > 0:
                self.writeV(node)
            if data.getTimeOrder() == 2:
                self.writeA(node)
            self.file.write('\n')
    def readOutput(file, timeStep = 0, node = 'all', val = 'u'):
        """
        This function read the ouput file that was produced by 
        this output class
        Input:
            file: file instance of output file
            timeStep:
                scalar: read the specific time step
                list, tuple: read the time steps in list or tuple
                'all': read all time steps
            node:
                scalar: read at specific node
                'all': read all nodes
            val:
                'u': read displacement u
                'v': read velocity v
                'a': read acceleration a
        Return:
            specific time step: numpy array
            other cases: list of numpy array
        """
        assert val != 'x','The output does not contain coordinates of nodes'
        return StandardFileOutput.readOutput(file,timeStep,node,val)
               