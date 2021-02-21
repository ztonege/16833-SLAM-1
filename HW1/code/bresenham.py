#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:33:27 2021

@author: ashwin
"""


import numpy as np
import random
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader

def bresenham(x1, y1, x2, y2, map_coord):
        
        dx = x2 - x1
        dy = y2 - y1
        
        slopedir = abs(dy) - abs(dx)
        
        if (slopedir > 0):
            x1, y1 = y1, x1
            x2, y2 = y2, x2
            
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
            
        dx = x2 - x1
        dy = y2 - y1
        
        error = int(dx /2.0)
        ystep = 1 if y1 < y2 else -1
        
        y = y1
        pointsx = []
        pointsy = []
        
        for x in range(x1, x2 + 1):
            
            if (slopedir > 0):
                coordx = y
                coordy = x
                
            else:
                coordx = x
                coordy = y
                
            pointsx.append(coordx)
            pointsy.append(coordy)
            
            ###########################
            m = map_coord[coordy, coordx]
            
            prob = random.choices([0,1], weights = (100*(1-m), 100*m), k = 1)
            # print(m, prob)
            
            if(prob == 1):
                distance = np.sqrt((coordx - x1)**2 + (coordy - y1)**2)
                return distance
                break
            else:
                continue
            ############################
                
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
                
        if swapped:
            pointsx.reverse()
            pointsy.reverse()
                
        return pointsx, pointsy


map_coord = np.array([[0.9, 0.1, 0.9, 0.4, 1],
             [0.6, 0.1, 0.9, 0.4, 1],
             [0.7, 0.1, 0.9, 0.4, 1],
             [0.8, 0.1, 0.9, 0.4, 1],
             [0.8, 0.1, 0.9, 0.4, 1]])
px, py = bresenham(0, 0, 3, 4, map_coord)
print(px)
print(py)