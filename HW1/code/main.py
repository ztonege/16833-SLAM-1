'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os


# from numba import jit, cuda

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
from multiprocessing import Pool


# @cuda.jit(target ="cuda")
def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    # plt.scatter(X_bar[:,0]/10.0, X_bar[:,1]/10.0)

    plt.imshow(occupancy_map, cmap='Greys')
    
    plt.axis([0, 800, 0, 800])

# @cuda.jit(target ="cuda")
def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o', s=5)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))   #REMOVE COMMENT
    plt.pause(0.00001)
    scat.remove()

# @cuda.jit(target ="cuda")
def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init

# @cuda.jit(target ="cuda")

def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    
    # X_bar_init = np.zeros((num_particles, 4))

    freespace_map = np.where(occupancy_map == 0)
    print("free space" ,np.shape(freespace_map))
    xfree, yfree = freespace_map
    print("min , max", np.min(xfree), np.max(xfree))
    index = np.random.randint(1, xfree.size, num_particles)
    
    y0_vals = 10.0*xfree[index].reshape(num_particles, 1)
    x0_vals = 10.0*yfree[index].reshape(num_particles, 1)
    
    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    
    return X_bar_init


def monte_carlo(mpargs):
    
    X_bar, num_particles = mpargs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()


   
    posex = 0
    posey = 0
    move = False
    firstpose = True
    
    first_time_idx = True
    for time_idx, line in enumerate(logfile):
        

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]
        
        if firstpose == True:
            posex = odometry_robot[0]
            posey = odometry_robot[1]
            firstpose = False
        
            
        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]
            nalla = True

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            
            X_bar[m, 0:3] = x_t1
            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
                # nalla = True
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
                # nalla = False

        X_bar = X_bar_new
        
        u_t0 = u_t1

        """
        RESAMPLING
        """
        currentposex = odometry_robot[0]
        currentposey = odometry_robot[1]
        
        if(currentposex - posex != 0 or currentposey - posey != 0):
            move = True
            X_bar = resampler.low_variance_sampler(X_bar)

        if move == True:
            posex = odometry_robot[0]
            posey = odometry_robot[1]
            move = False

            
        if nalla == True:
            visualize_timestep(X_bar, time_idx, args.output)

            visualize_map(occupancy_map)
            nalla = False
    # plt.scatter(X_bar[:,0], X_bar[:,1])


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=25, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    
    
    
    ##Multiprocessing
    
    mpargs = []
    mpargs.append((X_bar, num_particles))
    p = Pool()
    
    p.map(monte_carlo, mpargs)
  
            
    p.close()
    p.join()
      
        
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
# if args.visualize:
    # posex = 0
    # posey = 0
    # move = False
    # firstpose = True
       
    # first_time_idx = True
     
    # for time_idx, line in enumerate(logfile):
       
    #     # Read a single 'line' from the log file (can be either odometry or laser measurement)
    #     # L : laser scan measurement, O : odometry measurement
    #     meas_type = line[0]
       
    #     # convert measurement values from string to double
    #     meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
       
    #     # odometry reading [x, y, theta] in odometry frame
    #     odometry_robot = meas_vals[0:3]
    #     time_stamp = meas_vals[-1]
       
    #     if firstpose == True:
    #         posex = odometry_robot[0]
    #         posey = odometry_robot[1]
    #         firstpose = False
       
           
    #     # ignore pure odometry measurements for (faster debugging)
    #     # if ((time_stamp <= 0.0) | (meas_type == "O")):
    #     #     continue
       
    #     if (meas_type == "L"):
    #         # [x, y, theta] coordinates of laser in odometry frame
    #         odometry_laser = meas_vals[3:6]
    #         # 180 range measurement values from single laser scan
    #         ranges = meas_vals[6:-1]
    #         nalla = True
       
    #     print("Processing time step {} at time {}s".format(
    #         time_idx, time_stamp))
       
    #     if first_time_idx:
    #         u_t0 = odometry_robot
    #         first_time_idx = False
    #         continue
       
    #     X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
    #     u_t1 = odometry_robot
       
    #     # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
    #     # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
    #     for m in range(0, num_particles):
    #         """
    #         MOTION MODEL
    #         """
    #         x_t0 = X_bar[m, 0:3]
    #         x_t1 = motion_model.update(u_t0, u_t1, x_t0)
           
    #         X_bar[m, 0:3] = x_t1
    #         """
    #         SENSOR MODEL
    #         """
    #         if (meas_type == "L"):
    #             z_t = ranges
    #             w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
    #             X_bar_new[m, :] = np.hstack((x_t1, w_t))
    #             # nalla = True
    #         else:
    #             X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))
    #             # nalla = False
       
    #     X_bar = X_bar_new
       
    #     u_t0 = u_t1
       
    #     """
    #     RESAMPLING
    #     """
    #     currentposex = odometry_robot[0]
    #     currentposey = odometry_robot[1]
       
    #     if(currentposex - posex != 0 or currentposey - posey != 0):
    #         move = True
    #         X_bar = resampler.low_variance_sampler(X_bar)
       
    #     if move == True:
    #         posex = odometry_robot[0]
    #         posey = odometry_robot[1]
    #         move = False
       
           
    #     if nalla == True:
    #         visualize_timestep(X_bar, time_idx, args.output)
       
    #         visualize_map(occupancy_map)
    #         nalla = False
    # # plt.scatter(X_bar[:,0], X_bar[:,1])