import numpy as np
import matplotlib.pylab as plt
import os
import sys
import torch
import pickle
sys.path.append('/Users/yoonbyung/Dev/yet-another-mujoco-tutorial-v3/code/amp/poselib')
sys.path.append('/Users/yoonbyung/Dev/yet-another-mujoco-tutorial-v3/code/')
from collections import OrderedDict
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from util import t2pr
pkl_file_name =  "smplrig_cmu_walk_optimized_recon.pkl"
pkl_file_path = os.path.join(os.path.dirname(__file__), pkl_file_name)
with open(file=pkl_file_path, mode='rb') as f:
    pkl_file=pickle.load(f)
npy_file_name = 'amp_humanoid_run.npy'
npy_file = np.load(os.path.join(os.path.dirname(__file__), npy_file_name), allow_pickle=True)
skel_file = np.load(os.path.join(os.path.dirname(__file__), 'smpl_rig.npy'), allow_pickle=True)
# npy_file_name2 = 'post_process2.npy'
# npy_file2 = np.load(os.path.join(os.path.dirname(__file__), npy_file_name2), allow_pickle=True)
npy_file.item()['rotation']['arr'] = pkl_file['rotation'][:,1:,[1,2,3,0]]
# npy_file.item()['rotation']['arr'] = npy_file.item()['rotation']['arr'][:, :, [1,2,3,0]]
npy_file.item()['root_translation']['arr'] = np.array(pkl_file['p_root'])
npy_file.item()['skeleton_tree'] = skel_file.item()['skeleton_tree']#np.array(pkl_file['skel_local_translation'])
npy_file.item()['qpos'] = np.array(pkl_file['qpos'])
npy_file.item()['quat_joints'] = np.array(pkl_file['rotation'])
npy_file.item()['xpos'] = np.array(pkl_file['xpos'])
npy_file.item()['fps'] = 120
global_translation = torch.tensor(pkl_file['xpos'][:,1:,:])
global_rotation = torch.tensor(pkl_file['rotation'][:,1:,[1,2,3,0]])
global_velocity = SkeletonMotion._compute_velocity(p=global_translation, time_delta=1/120)
global_angular_velocity = SkeletonMotion._compute_angular_velocity(r=global_rotation, time_delta=1/120)
npy_file.item()['global_velocity']['arr'] = np.array(global_velocity)
npy_file.item()['global_angular_velocity']['arr'] = np.array(global_angular_velocity)
npy_file
pkl_file
# npy_file.item()['rotation']['arr'] = npy_file.item()['rotation']['arr'][:145]
# npy_file.item()['root_translation']['arr'] = npy_file.item()['root_translation']['arr'][:145]
# npy_file.item()['global_velocity']['arr'] = npy_file.item()['global_velocity']['arr'][:145]
# npy_file.item()['global_angular_velocity']['arr'] = npy_file.item()['global_angular_velocity']['arr'][:145]
# npy_file.item()['q_pos'] = npy_file.item()['q_pos'][:145]





