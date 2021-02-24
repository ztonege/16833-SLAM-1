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
from multiprocessing import Pool

from map_reader import MapReader
import time

# def bresenham(x1, y1, x2, y2, map_coord):
        
#         dx = x2 - x1
#         dy = y2 - y1
        
#         slopedir = abs(dy) - abs(dx)
        
#         if (slopedir > 0):
#             x1, y1 = y1, x1
#             x2, y2 = y2, x2
            
#         swapped = False
#         if x1 > x2:
#             x1, x2 = x2, x1
#             y1, y2 = y2, y1
#             swapped = True
            
#         dx = x2 - x1
#         dy = y2 - y1
        
#         error = int(dx /2.0)
#         ystep = 1 if y1 < y2 else -1
        
#         y = y1
#         pointsx = []
#         pointsy = []
        
#         for x in range(x1, x2 + 1):
            
#             if (slopedir > 0):
#                 coordx = y
#                 coordy = x
                
#             else:
#                 coordx = x
#                 coordy = y
                
#             pointsx.append(coordx)
#             pointsy.append(coordy)
            
#             ###########################
#             m = map_coord[coordy, coordx]
            
#             prob = random.choices([0,1], weights = (100*(1-m), 100*m), k = 1)
#             # print(m, prob)
            
#             if(prob == 1):
#                 distance = np.sqrt((coordx - x1)**2 + (coordy - y1)**2)
#                 return distance
#                 break
#             else:
#                 continue
#             ############################
                
#             error -= abs(dy)
#             if error < 0:
#                 y += ystep
#                 error += dx
                
#         if swapped:
#             pointsx.reverse()
#             pointsy.reverse()
                
#         return pointsx, pointsy

def bresenham(x1, y1, x2, y2):

        t1 = time.time()
        
        x1 = np.int(np.round(x1))
        y1 = np.int(np.round(y1))
        x2 = np.int(np.round(x2))
        y2 = np.int(np.round(y2))
        
        # x1 = np.int(x1)
        # y1 = np.int(y1)
        # x2 = np.int(x2)
        # y2 = np.int(y2)
        
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
        # pointsx = []
        points = []
        
        for x in range(x1, x2 + 1):
            
            if (slopedir > 0):
                coord = (y,x)
                # coord = x
                
            else:
                coord = (x,y)
                # coordy = y
                
            points.append(coord)
            # pointsy.append(coordy)
  
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
                
        if swapped:
            points.reverse()
            # pointsy.reverse()
        
        print("Time taken = ", time.time() - t1 )

        return points

map_coord = np.array([[0.9, 0.1, 0.9, 0.4, 1],
             [0.6, 0.1, 0.9, 0.4, 1],
             [0.7, 0.1, 0.9, 0.4, 1],
             [0.8, 0.1, 0.9, 0.4, 1],
             [0.8, 0.1, 0.9, 0.4, 1]])
px = bresenham(0, 0, 600, 200)
print(len(px))
# print(py)