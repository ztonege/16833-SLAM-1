'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import argparse
import transforms
import o3d_utility

from scipy.sparse import csc_matrix, eye, csr_matrix
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular


def find_projective_correspondence(source_points,
                                   source_normals,
                                   target_vertex_map,
                                   target_normal_map,
                                   intrinsic,
                                   T_init,
                                   dist_diff=0.07):
    '''
    \param source_points Source point cloud locations, (N, 3)
    \param source_normals Source point cloud normals, (N, 3)
    \param target_vertex_map Target vertex map, (H, W, 3)
    \param target_normal_map Target normal map, (H, W, 3)
    \param intrinsic Intrinsic matrix, (3, 3)
    \param T_init Initial transformation from source to target, (4, 4)
    \param dist_diff Distance difference threshold to filter correspondences
    \return source_indices: indices of points in the source point cloud with a valid projective correspondence in the target map, (M, 1)
    \return target_us: associated u coordinate of points in the target map, (M, 1)
    \return target_vs: associated v coordinate of points in the target map, (M, 1)
    '''
    h, w, _ = target_vertex_map.shape
    R = T_init[:3, :3]
    t = T_init[:3, 3:]

    # Transform source points from the source coordinate system to the target coordinate system
    T_source_points = (R @ source_points.T + t).T

    # Set up initial correspondences from source to target
    source_indices = np.arange(len(source_points)).astype(int)
    target_us, target_vs, target_ds = transforms.project(
        T_source_points, intrinsic)
    target_us = np.round(target_us).astype(int)
    target_vs = np.round(target_vs).astype(int)
    # print("h = ", h, "w =", w)
    # print("umin = ", np.min(target_us), "umax =", np.max(target_us))
    # TODO: first filter: valid projection
    mask = np.zeros_like(target_us).astype(bool)
    # print(np.shape(mask))
    mask1 = np.logical_and([target_us < w], [target_us >= 0]) 
    mask2 = np.logical_and([target_vs < h], [target_vs >= 0])
    mask3 = [target_ds >= 0]
    mask12 = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask12, mask3)
    mask = np.ravel(mask)

    # End of TODO
    # print("Len1 = ", len(target_us))
    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    # T_source_points = T_source_points[mask]
    # print("Len2 = ", len(target_us))
    # print("source indices = ", np.shape(source_indices))
    # print("target_us_max = ", np.max(target_us))
    # print("target_vs_max = ", np.max(target_vs))
    # # TODO: second filter: apply distance threshold
    mask = np.zeros_like(target_us).astype(bool)
    #print(source_indices)
    # print("Source shape = ", source_points.shape)
    # print("T_Source shape = ", T_source_points.shape)
    distx = (target_vertex_map[target_vs, target_us, 0] - T_source_points[source_indices, 0])
    disty = (target_vertex_map[target_vs, target_us, 1] - T_source_points[source_indices, 1])
    distz = (target_vertex_map[target_vs, target_us, 2] - T_source_points[source_indices, 2])
    dist = np.sqrt(distx**2 + disty**2 + distz**2)
    # print("Distance = ", dist)
    mask = [dist < dist_diff]
    # End of TODO

    source_indices = source_indices[mask]
    target_us = target_us[mask]
    target_vs = target_vs[mask]
    # print("Len3 = ", len(target_us))
    return source_indices, target_us, target_vs


def build_linear_system(source_points, target_points, target_normals, T):
    M = len(source_points)
    assert len(target_points) == M and len(target_normals) == M

    R = T[:3, :3]
    t = T[:3, 3:]

    p_prime = (R @ source_points.T + t).T
    q = target_points
    n_q = target_normals

    A = np.zeros((M, 6))
    b = np.zeros((M, ))

    # TODO: build the linear system
    # print("p_prime = ", p_prime.shape)
    # print("n_shape = ", n_q.shape)
    # print("q_shape = ", q.shape)

    A1 = n_q[:,2] * p_prime[:, 1] - n_q[:,1] * p_prime[:,2]
    A2 = n_q[:,0] * p_prime[:, 2] - n_q[:,2] * p_prime[:,0]
    A3 = n_q[:,1] * p_prime[:, 0] - n_q[:,0] * p_prime[:,1]
    A4 = n_q[:,0]
    A5 = n_q[:,1]
    A6 = n_q[:,2]

    A = np.vstack((A1, A2, A3, A4, A5, A6))
    A = A.T
    # print("A shape = ", A.shape)

    b = n_q[:,0] * (p_prime[:,0] - q[:,0]) + n_q[:,1] * (p_prime[:,1] - q[:,1]) + n_q[:,2] * (p_prime[:,2] - q[:,2])
    b = -b
    # b = b_og.reshape(-1, 1)
    # print("B shape = ", b.shape) 
    # End of TODO
    

    return A, b


def pose2transformation(delta):
    '''
    \param delta Vector (6, ) in the tangent space with the small angle assumption.
    \return T Matrix (4, 4) transformation matrix recovered from delta
    Reference: https://en.wikipedia.org/wiki/Euler_angles in the ZYX order
    '''
    w = delta[:3]
    u = np.expand_dims(delta[3:], axis=1)

    T = np.eye(4)

    # yapf: disable
    R = np.array([[
        np.cos(w[2]) * np.cos(w[1]),
        -np.sin(w[2]) * np.cos(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        np.sin(w[2]) * np.sin(w[0]) + np.cos(w[2]) * np.sin(w[1]) * np.cos(w[1])
    ],
    [
        np.sin(w[2]) * np.cos(w[1]),
        np.cos(w[2]) * np.cos(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.sin(w[0]),
        -np.cos(w[2]) * np.sin(w[0]) + np.sin(w[2]) * np.sin(w[1]) * np.cos(w[0])
    ],
    [
        -np.sin(w[1]),
        np.cos(w[1]) * np.sin(w[0]),
        np.cos(w[1]) * np.cos(w[0])
    ]])
    # yapf: enable

    T[:3, :3] = R
    T[:3, 3:] = u

    return T


def solve(A, b):
    '''
    \param A (6, 6) matrix in the LU formulation, or (N, 6) in the QR formulation
    \param b (6, 1) vector in the LU formulation, or (N, 1) in the QR formulation
    \return delta (6, ) vector by solving the linear system. You may directly use dense solvers from numpy.
    '''
    # TODO: write your relevant solver

    x = np.zeros((6, ))
    # U = eye(6)
    # lu = splu(A.T @ A, permc_spec= 'COLAMD')
    # x = lu.solve(A.T @ b)

    from scipy.sparse.linalg import spsolve
    x = spsolve(A.T @ A, A.T @ b)

    # x = inv(A) @ b
    return x


def icp(source_points,
        source_normals,
        target_vertex_map,
        target_normal_map,
        intrinsic,
        T_init=np.eye(4),
        debug_association=False):
    '''
    \param source_points Source point cloud locations, (N, 3)
    \param source_normals Source point cloud normals, (N, 3)
    \param target_vertex_map Target vertex map, (H, W, 3)
    \param target_normal_map Target normal map, (H, W, 3)
    \param intrinsic Intrinsic matrix, (3, 3)
    \param T_init Initial transformation from source to target, (4, 4)
    \param debug_assocation Visualize association between sources and targets for debug
    \return T (4, 4) transformation from source to target
    '''

    T = T_init

    for i in range(70):
        print("iteration = ", i)
        # TODO: fill in find_projective_correspondences
        source_indices, target_us, target_vs = find_projective_correspondence(
            source_points, source_normals, target_vertex_map,
            target_normal_map, intrinsic, T)

        # Select associated source and target points
        corres_source_points = source_points[source_indices]
        corres_target_points = target_vertex_map[target_vs, target_us]
        corres_target_normals = target_normal_map[target_vs, target_us]

        # Debug, if necessary
        if debug_association:
            o3d_utility.visualize_correspondences(corres_source_points,
                                                  corres_target_points, T)

        # TODO: fill in build_linear_system and solve
        A, b = build_linear_system(corres_source_points, corres_target_points,
                                   corres_target_normals, T)
        delta = solve(A, b)

        # Update and output
        T = pose2transformation(delta) @ T
        loss = np.mean(b**2)
        print('iter {}: avg loss = {:.4e}, inlier count = {}'.format(
            i, loss, len(corres_source_points)))

    return T


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--source_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=10)
    parser.add_argument('--target_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=100)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)

    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    # TUM convention -- uint16 value to float meters
    depth_scale = 5000.0

    # Source: load depth and rescale to meters
    source_depth = o3d.io.read_image('{}/{}.png'.format(
        depth_path, args.source_idx))
    source_depth = np.asarray(source_depth) / depth_scale

    # Unproject depth to vertex map (H, W, 3) and reshape to a point cloud (H*W, 3)
    source_vertex_map = transforms.unproject(source_depth, intrinsic)
    source_points = source_vertex_map.reshape((-1, 3))

    # Load normal map (H, W, 3) and reshape to point cloud normals (H*W, 3)
    source_normal_map = np.load('{}/{}.npy'.format(normal_path,
                                                   args.source_idx))
    source_normals = source_normal_map.reshape((-1, 3))

    # Similar preparation for target, but keep the image format for projective association
    target_depth = o3d.io.read_image('{}/{}.png'.format(
        depth_path, args.target_idx))
    target_depth = np.asarray(target_depth) / depth_scale
    target_vertex_map = transforms.unproject(target_depth, intrinsic)
    target_normal_map = np.load('{}/{}.npy'.format(normal_path,
                                                   args.target_idx))

    # Visualize before ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape(
        (-1, 3)), np.eye(4))

    # TODO: fill-in components in ICP
    T = icp(source_points,
            source_normals,
            target_vertex_map,
            target_normal_map,
            intrinsic,
            np.eye(4),
            debug_association=False)

    # Visualize after ICP
    o3d_utility.visualize_icp(source_points, target_vertex_map.reshape(
        (-1, 3)), T)
