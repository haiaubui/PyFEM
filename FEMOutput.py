# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 17:52:32 2017

@author: haiau
"""

import sys
import numpy as np
import FEMMesh as fm
import warnings

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
        
class HarmonicsOutput(FileOutput):
    """
    Output harmonics paramereters to file
    """
    def outputData(self, data):
        """
        write data to output file
        """
        try:
            if data.mesh.nHarmonic < 0:
                return
        except AttributeError:
            return
        outstr = 'Time '+str(data.getTime())+' NH '+str(data.mesh.nHarmonic)
        self.file.write(outstr+'\n')
        for i in range(data.mesh.nHarmonic):
            outstr = str(data.U[data.mesh.harmonicsIDa[i]])+' '
            outstr += str(data.U[data.mesh.harmonicsIDb[i]])+' '
            outstr += str(data.U[data.mesh.harmonicsIDc[i]])+' '
            outstr += str(data.U[data.mesh.harmonicsIDd[i]])+'\n'
            self.file.write(outstr)
    
    @staticmethod
    def __read_hdr__(file):
        hdr = file.readline()
        ex = Exception('The file is corrupted or false formatted!')
        if hdr == '':
            raise EOFError
        hdr = hdr.split()
        if hdr[0] != 'Time':
            raise ex
        try:
            time = float(hdr[1])
        except Exception:
            raise ex
        if hdr[2] != 'NH':
            raise ex
        try:
            numHar = int(hdr[3])
        except Exception:
            raise ex
        return time,numHar
    
    @staticmethod
    def __nextStep__(file, step, prev = -1):
        for i in range(step-prev-1):
            time,numHar = HarmonicsOutput.__read_hdr__(file)
            for har in range(numHar):
                file.readline()
            
    
    @staticmethod
    def __readOneStep__(file):
        time,numHar = HarmonicsOutput.__read_hdr__(file)
        res = []
        for har in range(numHar):
            try:
                dat = file.readline().split()
            except AttributeError:
                break
            res.append(np.array([float(a) for a in dat]))
        return time, res
    
    @staticmethod        
    def readOutput(filen, timeStep = 0):
        """
        This function read the ouput file that was produced by 
        this output class
        Input:
            filen: name of output file
            timeStep:
                scalar: read the specific time step
                list, tuple: read the time steps in list or tuple
                'all': read all time steps
        Return:
            list of numpy array
        """
        __max_allowed_steps__ = 1000000
        file = open(filen,'r')
            
        alltime = True and timeStep == 'all'
        res = []
        time = []
        if alltime:
            i = 0
            while i < __max_allowed_steps__:
                i += 1
                try:
                    t,r = HarmonicsOutput.__readOneStep__(file)
                    time.append(t)
                    res.append(r)
                except EOFError:
                    file.close()
                    return time, res
            file.close()
            return time,res
        else:
            asc = isAscending(timeStep)
            prev = -1
            for step in timeStep:
                if not asc:
                    file.seek(0)
                try:
                    HarmonicsOutput.__nextStep__(file,step,prev)
                    t,r = HarmonicsOutput.__readOneStep__(file)
                except EOFError:
                    file.close()
                    return time, res
                time.append(t)
                res.append(r)
                if asc:
                    prev = step
            file.close()
            return time, res
                
                
    
            
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
    def chooseSteps(self, steps = 'all'):
        """
        Choose time steps to be output
        steps = 'all' : output all steps
        otherwise: output all in list, tuple, iterator, generator steps
        """
        if steps == 'all':
            self.allsteps = True
        else:
            self.allsteps = False
            self.steps = steps
            
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
            try:
                self.file.write(' LOAD '+str(data.getLambda())+' ORDER '+\
                                str(data.getTimeOrder())+'\n')
            except AttributeError:
                self.file.write(' TIME 0.0 ORDER 0\n')
    
    def outputData(self, data):
        """
        write data to output file
        """
        try:
            if not self.allsteps:
                if data.getCurrentStep() not in self.steps:
                    return
        except:
            self.allsteps = True
                
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
    
    @staticmethod    
    def readHeader(file):
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
        if hdr[2] != 'TIME' and hdr[2] != 'LOAD':
            raise ex
        try:
            t = float(hdr[3])
        except:
            raise ex
        if hdr[4] != 'ORDER':
            raise ex
        try:
            torder = int(hdr[5])
        except:
            raise ex
            
        return Nnod, t, torder
            
    @staticmethod
    def __readNodes(file, node, val, Nnod):
        res = []
        allnode = True and isinstance(node, str)
        somenode = not allnode and isinstance(node, (tuple, list))
        
        for i in range(Nnod):            
            line = file.readline()
            if line is None:
                raise EOFError
            if not allnode:
                if not somenode:
                    if node != i:
                        continue
                else:
                    if i not in node:
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
                    if len(v) == 0:
                        file.close()
                        raise Exception
                    res.append(v)
                elif val == 'a':
                    a = list(float(x) for x in line[4+Ndof*2:4+Ndof*3])
                    if len(a) == 0:
                        file.close()
                        raise Exception
                    res.append(a)
                elif val == 'all':
                    al = list(float(x) for x in line[1:-1])
                    res.append(al)
                else:
                    file.close()
                    raise Exception('Cannot read '+val+' in this file')
            except IndexError:
                file.close()
                raise Exception('Cannot read '+val+' in this file')
        return np.array(res)       
            
    @staticmethod        
    def readOutput(filen, timeStep = 0, node = 'all', val = 'u'):
        """
        This function read the ouput file that was produced by 
        this output class
        Input:
            filen: name of output file
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
                'all': everything
        Return:
            list of numpy array
        """
        file = open(filen,'r')
        Nnod, ti, timeOrder = StandardFileOutput.readHeader(file)
        assert not(isinstance(timeStep,str) and timeStep != 'all'),\
        'unknown option '+ str(timeStep)
            
        alltime = True and timeStep == 'all'
        res = []
        if not alltime:
            if isinstance(timeStep,(list,tuple)):
                rest = []
                for i in timeStep:
                    try:
                        a,t = StandardFileOutput.readOutput(filen,i,node,val)
                        res.append(a)
                        rest.append(t)
                    except EOFError:
                        warnings.warn('file ended before all steps were read')
                        break
                file.close()
                return res,rest
            else:
                gotoLine(file, (Nnod+1)*timeStep)
                _,t,_ = StandardFileOutput.readHeader(file)
                return StandardFileOutput.__readNodes(file,node,val,Nnod),t
        try:
            rest = []
            file.seek(0)
            while 1:
               _,t,_ = StandardFileOutput.readHeader(file)
               rest.append(t)
               res.append(StandardFileOutput.__readNodes(file,node,val,Nnod))
        except EOFError:
            file.close()
            return res,rest
            
    def updateToMesh(self, mesh, istep = 0):
        """
        update output data to mesh at a specific time step
        """
        StandardFileOutput.updateMesh(self.outfile,mesh,istep)
    
    @staticmethod            
    def updateMesh(file, mesh, istep = 0):
        """
        update output data to mesh at a specific time step
        """
        resu,_ = StandardFileOutput.readOutput(file,istep)
        try:
            resv,_ = StandardFileOutput.readOutput(file,istep,val='v')
        except Exception:
            resv = None
        try:
            resa,_ = StandardFileOutput.readOutput(file,istep,val='a')
        except Exception:
            resa = None
        nodes = mesh.getNodes()
        for i,n in enumerate(nodes):
            n.setU(resu[i])
        if resv is not None:
            for i,n in enumerate(nodes):
                n.setV(resv[i])
        if resa is not None:
            for i,n in enumerate(nodes):
                try:
                    n.setA(resa[i])
                except ValueError:
                    print('error here')
                    raise ValueError
    
    def getValueInElement(self, mesh, e, x, isteps = 0, val='u'):
        """
        get output value at position x(natural coordinate) in element e of mesh
        """
        IT = fm.get_connection(mesh,e)
        try:
            resuu = []
            for i in isteps:
                resu,tout =StandardFileOutput.readOutput(self.outfile,i,IT,val)
                for j,n in enumerate(e.getNodes()):
                    n.setU(resu[i][j])
                resuu.append(e.values_at(x,val))
            return resuu
        except Exception:
            resu,tout =StandardFileOutput.readOutput(self.outfile,i,IT,val)
            for j,n in enumerate(e.getNodes()):
                n.setU(resu[i][j])
            return e.values_at(x,val)
                
            

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
            
def isAscending(ar):
    """
    check if array is ascending ordered
    """
    for i,a in enumerate(ar):
        try:
            if a > ar[i+1]:
                return False
        except IndexError:
            return True
    return True
 
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
    
    def __readNodes(file, node, val, Nnod):
        res = []
        allnode = True and isinstance(node, str)
        somenode = not allnode and isinstance(node, (tuple, list))
        
        for i in range(Nnod):            
            line = file.readline()
            if line is None:
                raise EOFError
            if not allnode:
                if not somenode:
                    if node != i:
                        continue
                else:
                    if i not in node:
                        continue
            line = line.split()
            Ndof = int(line[0])
            try:
                if val == 'u':
                    u = list(float(x) for x in line[1:1+Ndof])
                    res.append(u)
                elif val == 'v':
                    v = list(float(x) for x in line[1+Ndof:1+Ndof*2])
                    res.append(v)
                elif val == 'a':
                    a = list(float(x) for x in line[1+Ndof*2:1+Ndof*3])
                    res.append(a)
                elif val == 'all':
                    al = list(float(x) for x in line[1:-1])
                    res.append(al)
                else:
                    file.close()
                    raise Exception('Cannot read '+val+' in this file')
            except IndexError:
                file.close()
                raise Exception('Cannot read '+val+' in this file')
        return np.array(res)  
               