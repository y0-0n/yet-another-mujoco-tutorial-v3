# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from enum import Enum
import numpy as np
from torch import Tensor
import torch
import os
import ray

from gym import spaces

# from isaacgym import gymapi
# from isaacgym import gymtorch

from .amp.atlas_amp_base import AtlasAMPBase, dof_to_obs
# from .amp.utils_amp import gym_util
from amp.utils.motion_lib import MotionLib

# from isaacgym.torch_utils import *
# from isaacgymenvs.utils.torch_jit_utils import *
from amp.utils.amp_torch_utils import *

# modified for Atlas
NUM_AMP_OBS_PER_STEP = 85 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos(6d), dof_vel, key_body_pos]
NUM_DEEPMIMIC_OBS = 85 # [root_p+root_rot+dof_pos(77), root_vel+root_ang_vel+dof_vel(43), key_body_pos(12), COM(3)]

class AtlasAMP(AtlasAMPBase):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = AtlasAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []


        # motion_file = cfg['env'].get('motion_file')
        # motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../asset/" + motion_file)
        # self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        # self._deepmimic_obs_buf = torch.zeros((self.num_envs, NUM_DEEPMIMIC_OBS), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
        # sample motion
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return
        

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device='cpu')
                                    #  device=self.device)
        self._motion_lib_demo = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    
    # yoon0-0
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, collision_test=True):
        # set_envs = []
        for i, env_id in enumerate(env_ids):
            q=np.concatenate((root_pos[i].detach().cpu().numpy(), root_rot[i].cpu().numpy(), dof_pos[i].detach().cpu().numpy()),axis=0) # root_pos, root_rot, dof_pos
            dq=np.concatenate((root_vel[i].detach().cpu().numpy(), root_ang_vel[i].detach().cpu().numpy(), dof_vel[i].detach().cpu().numpy()),axis=0) # root_pos, root_rot, dof_pos
            self.mujoco_envs[env_id].assign_vel.remote(dq=dq,joint_idxs=list(range(0,6))+self.standard_env.ctrl_qvel_idxs)
            self.mujoco_envs[env_id].forward.remote(q=q,joint_idxs=list(range(0,7))+self.standard_env.ctrl_qpos_idxs,INCREASE_TICK=True)
        # q=np.concatenate((np.array(root_pos), np.array(root_rot[:, [3,0,1,2]]), np.array(dof_pos)),axis=-1) # root_pos, root_rot, dof_pos
        # dq=np.concatenate((np.array(root_vel), np.array(root_ang_vel), np.array(dof_vel)),axis=-1) # root_pos, root_rot, dof_pos
        # ray.get(set_envs)
        # for i in env_ids:
            
        #     self.mujoco_envs[i].forward.remote(q=q,joint_idxs=list(range(0,7))+(self.standard_env.rev_joint_idxs+6).tolist(),INCREASE_TICK=False)
        return
    
    def _reset_soccer_ball(self, env_ids):
        variance_soccer_ball_position = 0.5
        env_soccer_ball_ids_int32 = self.soccer_ball_id[env_ids].to(dtype=torch.int32)
        self._root_states[self.soccer_ball_id[env_ids], :] = self._ball_buffer[self.soccer_ball_id[env_ids], :] # initial root poses
        self._root_states[self.soccer_ball_id[env_ids], 1] = torch.randn(len(env_ids)).to(self.device) * variance_soccer_ball_position
        # reference
        #self._ball_buffer[env_ids,0] = 4.0*torch.rand(len(env_ids), device=self.device) + 2.5
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                gymtorch.unwrap_tensor(env_soccer_ball_ids_int32), len(env_soccer_ball_ids_int32))

        return env_soccer_ball_ids_int32

    # Added from JTM
    def _reset_balls(self, env_ids):
        env_ball_ids_int32 = self.ball_ids[env_ids].to(dtype=torch.int32)
        self._root_states[self.ball_ids[env_ids], :] = self._ball_buffer[self.ball_ids[env_ids], :]
        # reference
        #self._ball_buffer[env_ids,0] = 4.0*torch.rand(len(env_ids), device=self.device) + 2.5
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ball_ids_int32), len(env_ball_ids_int32))
    # Added from JTM
    def _reset_boxs(self, env_ids):
        env_ball_ids_int32 = self.box_ids[env_ids].to(dtype=torch.int32)
        self._root_states[self.box_ids[env_ids], :] = self._box_buffer[self.box_ids[env_ids], :]
        # reference
        #self._ball_buffer[env_ids,0] = 4.0*torch.rand(len(env_ids), device=self.device) + 2.5
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ball_ids_int32), len(env_ball_ids_int32))

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self._num_amp_obs_steps - 1)]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self._root_states[self.humanoid_ids], self._dof_pos, self._dof_vel, key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._root_states[self.humanoid_ids[env_ids]], self._dof_pos[env_ids], 
                                                                    self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_pos, dof_vel, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def build_deepmimic_observations(root_states, dof_pos, dof_vel, key_body_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_rot_obs = root_rot

    # dof_obs = dof_to_obs(dof_pos)    

    key_body_pos = key_body_pos - root_pos.unsqueeze(-2)
    flat_local_key_pos = key_body_pos.view(key_body_pos.shape[0], key_body_pos.shape[1] * key_body_pos.shape[2])

    obs = torch.cat((root_pos, root_rot_obs, root_vel, root_ang_vel, dof_pos, dof_vel, flat_local_key_pos), dim=-1)
    return obs