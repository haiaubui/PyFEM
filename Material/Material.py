# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:57:14 2017

@author: haiau
"""

class Material(object):
    """
    Material class stores all properties and methods of an material
    """
    def calculate(self, data = None):
        """
        calculate material parameters. These parameters is stored in this 
        material object.
        Input:
            data: optional, object class that can return required value for
            calculation. It is normally an Element object
        """
        pass
    
    def getID(self):
        """
        Get id number of this materal
        """
        return 0