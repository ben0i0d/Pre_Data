"""
Utility script containing pre-processing logic.
Main call is pre_normalization: 
    Pads empty frames, 
    Centers human, 
    Align joints to axes.
"""

import math
import numba
import psutil
import numpy as np
from tqdm import tqdm
from joblib import Parallel , delayed

def pre_normalization(data, zaxis=[11, 5]):
    """
    Normalization steps:
        1) Rotate human to align specified joints to z-axis: ntu [0,1], uav [11,5]
    
    Args:
        data: tensor with skeleton data of shape N, M, T, V, C
        zaxis: list containing 0 or 2 body joint indices (0 skips the alignment)
    """
    def align_human_to_vector(i_s, skeleton, joint_idx1: int, joint_idx2: int, target_vector: list):
        joint1 = skeleton[0, 0, joint_idx1]
        joint2 = skeleton[0, 0, joint_idx2]
        axis = np.cross(joint2 - joint1, target_vector)
        angle = angle_between(joint2 - joint1, target_vector)
        matrix = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    data[i_s, i_p, i_f, i_j] = np.dot(matrix, joint)
    # uav parallel the bone between hip(jpt 11)and spine(jpt 5) of the first person to the z axis
    print('parallel the bone between hip(jpt %s)' %zaxis[0] + 'and spine(jpt %s) of the first person to the z axis' %zaxis[1])
    Parallel(n_jobs=psutil.cpu_count(logical=False), verbose=0)(delayed(lambda i,s: align_human_to_vector(i,s,zaxis[0], zaxis[1], [0, 0, 1]))(i,s) for i,s in enumerate(tqdm(data)))
    return data

@numba.jit(nopython=True)
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    """ Returns the unit vector of the vector.  """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

@numba.jit(nopython=True)
def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)

@numba.jit(nopython=True)
def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)

@numba.jit(nopython=True)
def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)