'''
    Initially written by Ming Hsiao in MATLAB
    Redesigned and rewritten by Wei Dong (weidong@andrew.cmu.edu)
'''
import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import quaternion  # pip install numpy-quaternion

import transforms
import o3d_utility

from preprocess import load_gt_poses


class Map:
    def __init__(self):
        self.points = np.empty((0, 3))
        self.normals = np.empty((0, 3))
        self.colors = np.empty((0, 3))
        self.weights = np.empty((0, 1))
        self.initialized = False

    def merge(self, indices, points, normals, colors, R, t):
        '''
        TODO: implement the merge function
        \param self The current maintained map
        \param indices Indices of selected points. Used for IN PLACE modification.
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \return None, update map properties IN PLACE
        '''
        self.points[indices, :] = (self.weights[indices, :]*self.points[indices, :] + (R @ points.T + t).T)/(self.weights[indices, :] + 1)   
        self.normals[indices, :] = (self.weights[indices, :]*self.normals[indices, :] + (R @ normals.T + t).T)/(self.weights[indices, :] + 1)
        self.normals[indices, :] = self.normals[indices, :]/np.linalg.norm(self.normals[indices, :])
        self.colors[indices, :] = (self.weights[indices, :]*self.colors[indices, :] + colors)/(self.weights[indices, :] + 1)
        self.weights[indices, :] = self.weights[indices, :] + 1
        
    def add(self, points, normals, colors, R, t):
        '''
        TODO: implement the add function
        \param self The current maintained map
        \param points Input associated points, (N, 3)
        \param normals Input associated normals, (N, 3)
        \param colors Input associated colors, (N, 3)
        \param R rotation from camera (input) to world (map), (3, 3)
        \param t translation from camera (input) to world (map), (3, )
        \return None, update map properties by concatenation
        '''
        # self.points = points
        # self.normals = normals
        # self.colors = colors
        # if not self.initialized:
        # self.weights = np.ones(len(points))

        self.weights = np.concatenate((self.weights, np.ones((points.shape[0],1))))
        self.points = np.concatenate((self.points, (R @ points.T + t).T))
        self.normals = np.concatenate((self.normals, (R @ normals.T + t).T))
        self.colors = np.concatenate((self.colors, colors))
        

    def filter_pass1(self, us, vs, ds, h, w):
        '''
        TODO: implement the filter function
        \param self The current maintained map, unused
        \param us Putative corresponding u coordinates on an image, (N, 1)
        \param vs Putative corresponding v coordinates on an image, (N, 1)
        \param vs Putative corresponding d depth on an image, (N, 1)
        \param h Height of the image projected to
        \param w Width of the image projected to
        \return mask (N, 1) in bool indicating the valid coordinates
        '''
        mask = np.zeros_like(us).astype(bool)
        # print(np.shape(mask))
        mask1 = np.logical_and([us < w], [us >= 0]) 
        mask2 = np.logical_and([vs < h], [vs >= 0])
        mask3 = [ds >= 0]
        mask12 = np.logical_and(mask1, mask2)
        mask = np.logical_and(mask12, mask3)
        mask = mask.T
        mask = np.ravel(mask)
        return mask
        # return np.zeros_like(us)

    def filter_pass2(self, points, normals, input_points, input_normals,
                     dist_diff, angle_diff):
        '''
        TODO: implement the filter function
        \param self The current maintained map, unused
        \param points Maintained associated points, (M, 3)
        \param normals Maintained associated normals, (M, 3)
        \param input_points Input associated points, (M, 3)
        \param input_normals Input associated normals, (M, 3)
        \param dist_diff Distance difference threshold to filter correspondences by positions
        \param angle_diff Angle difference threshold to filter correspondences by normals
        \return mask (N, 1) in bool indicating the valid correspondences
        '''
        indices = np.arange(len(points)).astype(int)


        mask = np.zeros_like(indices).astype(bool)
        distx = (points[indices, 0] - input_points[indices, 0])
        disty = (points[indices, 1] - input_points[indices, 1])
        distz = (points[indices, 2] - input_points[indices, 2])
        dist = np.sqrt(distx**2 + disty**2 + distz**2)
        # print("Distance = ", dist)
        mask1 = [dist < dist_diff]
        # print("Normals = ", normals[0,0])
        # print("Points = ", points[0,0])
        n_points = normals - points
        n_input_points = input_normals - input_points
        # print("n_points shape = ", n_points.shape)
        # print("n_input_points shape = ", n_input_points.shape)
        mag_normals = np.linalg.norm(normals, axis = 1)
        mag_normals = mag_normals.reshape(-1,1).repeat(normals.shape[1], axis = 1)
        # print("SHAPE = ", mag.shape)
        unit_normal = normals / mag_normals


        mag_input_normal = np.linalg.norm(input_normals, axis = 1)
        mag_input_normal = mag_input_normal.reshape(-1,1).repeat(input_normals.shape[1], axis = 1)
        unit_input_normal = input_normals / mag_input_normal


        angle = np.sum(np.multiply(unit_normal, unit_input_normal), axis = 1)
        # angle = np.sum(unit_normal*unit_input_normal, axis = 1)
        # i = [angle > np.pi]
        # angle
        # print("Angle Shape = ", angle.shape)
        angle = np.abs(np.arccos(angle))
        # print(angle)
        mask2 = [angle < angle_diff]

        mask = np.logical_and(mask1, mask2)
        mask = np.ravel(mask)
        return mask
        # return np.zeros((len(points)))

    def fuse(self,
             vertex_map,
             normal_map,
             color_map,
             intrinsic,
             T,
             dist_diff=0.03,
             angle_diff=np.deg2rad(5)):
        '''
        \param self The current maintained map
        \param vertex_map Input vertex map, (H, W, 3)
        \param normal_map Input normal map, (H, W, 3)
        \param intrinsic Intrinsic matrix, (3, 3)
        \param T transformation from camera (input) to world (map), (4, 4)
        \return None, update map properties on demand
        '''
        # Camera to world
        R = T[:3, :3]
        t = T[:3, 3:]

        # World to camera
        T_inv = np.linalg.inv(T)
        R_inv = T_inv[:3, :3]
        t_inv = T_inv[:3, 3:]

        if not self.initialized:
            points = vertex_map.reshape((-1, 3)) #points in image frame
            normals = normal_map.reshape((-1, 3)) 
            colors = color_map.reshape((-1, 3))

            # TODO: add step
            self.add(points, normals, colors, R, t)
            self.initialized = True

        else:    
            h, w, _ = vertex_map.shape #coming from next image, to which we project the world model

            # Transform from world to camera for projective association
            indices = np.arange(len(self.points)).astype(int) #indices of points in the world that is maintained
            T_points = (R_inv @ self.points.T + t_inv).T #Transformed points in camera frame
            R_normals = (R_inv @ self.normals.T).T # Transformed normals in camera frame

            # Projective association
            us, vs, ds = transforms.project(T_points, intrinsic) #pixel locations in camera of world model
            us = np.round(us).astype(int)
            vs = np.round(vs).astype(int)

            # TODO: first filter: valid projection
            mask = self.filter_pass1(us, vs, ds, h, w)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask]

            T_points = T_points[indices]
            R_normals = R_normals[indices]
            valid_points = vertex_map[vs, us] #Need to update this stuff
            valid_normals = normal_map[vs, us]

            # TODO: second filter: apply thresholds
            mask = self.filter_pass2(T_points, R_normals, valid_points,
                                     valid_normals, dist_diff, angle_diff)
            # Should not happen -- placeholder before implementation
            if mask.sum() == 0:
                return
            # End of TODO

            indices = indices[mask]
            us = us[mask]
            vs = vs[mask] #Pixel locations of the world map that need to be updated from merging with current frame

            updated_entries = len(indices)

            merged_points = vertex_map[vs, us] #These points will be merged with the corresponding points in global model
            merged_normals = normal_map[vs, us]
            merged_colors = color_map[vs, us]

            # TODO: Merge step - compute weight average after transformation
            self.merge(indices, merged_points, merged_normals, merged_colors,
                       R, t)
            # End of TODO

            associated_mask = np.zeros((h, w)).astype(bool)
            associated_mask[vs, us] = True #Have been merged
            new_points = vertex_map[~associated_mask] #Points in current frame that are not yet meregd with global model
            new_normals = normal_map[~associated_mask]
            new_colors = color_map[~associated_mask]

            # TODO: Add step
            self.add(new_points, new_normals, new_colors, R, t)
            # End of TODO

            added_entries = len(new_points)
            print('updated: {}, added: {}, total: {}'.format(
                updated_entries, added_entries, len(self.points)))
            # print("h, w =", h, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'path', help='path to the dataset folder containing rgb/ and depth/')
    parser.add_argument('--start_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=1)
    parser.add_argument('--end_idx',
                        type=int,
                        help='index to the source depth/normal maps',
                        default=200)
    parser.add_argument('--downsample_factor', type=int, default=2)
    args = parser.parse_args()

    intrinsic_struct = o3d.io.read_pinhole_camera_intrinsic('intrinsics.json')
    intrinsic = np.array(intrinsic_struct.intrinsic_matrix)
    indices, gt_poses = load_gt_poses(
        os.path.join(args.path, 'livingRoom2.gt.freiburg'))
    # TUM convention
    depth_scale = 5000.0

    rgb_path = os.path.join(args.path, 'rgb')
    depth_path = os.path.join(args.path, 'depth')
    normal_path = os.path.join(args.path, 'normal')

    m = Map()
    
    down_factor = args.downsample_factor
    intrinsic /= down_factor
    intrinsic[2, 2] = 1


    for i in range(args.start_idx, args.end_idx + 1):
        print('Fusing frame {:03d}'.format(i))
        source_depth = o3d.io.read_image('{}/{}.png'.format(depth_path, i))
        source_depth = np.asarray(source_depth) / depth_scale
        source_depth = source_depth[::down_factor, ::down_factor]
        source_vertex_map = transforms.unproject(source_depth, intrinsic)
        h, w, _ = source_vertex_map.shape
        # print("h, w =", h, w)

        source_color_map = np.asarray(
            o3d.io.read_image('{}/{}.png'.format(rgb_path,
                                                 i))).astype(float) / 255.0
        source_color_map = source_color_map[::down_factor, ::down_factor]

        source_normal_map = np.load('{}/{}.npy'.format(normal_path, i))
        source_normal_map = source_normal_map[::down_factor, ::down_factor]

        m.fuse(source_vertex_map, source_normal_map, source_color_map,
               intrinsic, gt_poses[i])

    global_pcd = o3d_utility.make_point_cloud(m.points,
                                              colors=m.colors,
                                              normals=m.normals)
    o3d.visualization.draw_geometries(
        [global_pcd.transform(o3d_utility.flip_transform)])

    totalpoints = 640*480*200
    mappoints = m.points.shape[0]
    print("Number of points = ", mappoints)
    print("Total number of points = ", totalpoints)
    compression = mappoints/totalpoints
    print("Compression Ratio = ", compression)

    