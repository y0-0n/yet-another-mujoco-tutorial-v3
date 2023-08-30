import os,cv2
import numpy as np
import mujoco
import mujoco_viewer
from util import pr2t,r2w,rpy2r,trim_scale,meters2xyz,compute_view_params
import ray
import time
from mujoco_parser import MuJoCoParserClass
import torch
from util import *
from pid import PID_ControllerClass

@ray.remote
class MuJoCoParserClassRay(MuJoCoParserClass):
    """
        MuJoCo Parser class
    """
    def __init__(self,name='Robot',rel_xml_path=None,USE_MUJOCO_VIEWER=False,VERBOSE=True, env_id=0):
        """
            Initialize MuJoCo parser with ray
        """
        super().__init__(name=name,rel_xml_path=rel_xml_path,USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER,VERBOSE=VERBOSE)

        self.PID = PID_ControllerClass(
                name = 'PID',dim = self.n_ctrl,
                k_p = 1.5, k_i = 0.01, k_d = 0.001,
                out_min = self.ctrl_ranges[:,0],
                out_max = self.ctrl_ranges[:,1],
                ANTIWU  = True)
        
        # Change floor friction
        self.model.geom('floor').friction = np.array([1,0.01,0]) # default: np.array([1,0.01,0])
        self.model.geom('floor').priority = 1 # >0
        print('floor friction')
    
    def play_steps(self):
        raise NotImplementedError("play_steps not implemented")
        #for n in range(self.horizon_length):
        # for n in range(16):
        #     self._

    # ______________________ AMP utils ______________________

    def reset_PID(self):
        self.PID.reset()

    def assign_vel(self,dq=None,joint_idxs=None):
        if dq is not None:
            if joint_idxs is not None:
                self.data.qvel[joint_idxs] = dq
            else:
                self.data.qvel = dq
        # mujoco.mj_forward(self.model,self.data)
        # if INCREASE_TICK:
        #     self.tick = self.tick + 1

    # def get_v_base(self):
    #     """
    #         Get body position
    #     """
    #     return self.data.qvel[0:3].copy()

    def get_ps(self):
        """
            Get x
        """
        return self.data.xpos[1:].copy()

    def get_qposes(self):
        """
            Get 
        """
        return self.data.qpos[self.rev_joint_idxs+6].copy()

    def get_qvels(self):
        """
            Get 
        """
        return self.data.qvel[self.rev_joint_idxs+5].copy()
    
    # _______________________________________________________
    def generator_step(self, action, render_every=1):
        return
    
    def step_queue(self, ctrl, queue):
        """
            Step
        """
        if self.is_running:
            return
        
        print("queue value: ", r)

        self.is_running = True
        
        self.step(ctrl=ctrl)
        p, R = self.get_pR_body()
        queue.put_async({"position": p, "rotation": R})

        self.is_running = False

    def step(self,ctrl=None,ctrl_idxs=None,nstep=1,INCREASE_TICK=True):
        """
            Step with PD Controller
        """

        super().step(ctrl=ctrl,ctrl_idxs=ctrl_idxs,nstep=nstep,INCREASE_TICK=INCREASE_TICK)

        # actor_root_states = position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

        result_dict = {
            "actor_root_states" : np.concatenate((self.get_p_body('base'), r2quat(self.get_R_body('base'))[[1,2,3,0]], self.get_qvel_joint('base')[0:3], self.get_qvel_joint('base')[3:6]), axis=-1),
            "dof_pos": self.get_qposes(),
            "dof_vel": self.get_qvels(),
            "rigid_body_pos": self.get_ps(),
            "contact_info": self.get_contact_info()
            # "root_p": self.get_p_body('base'),
            # "root_R": self.get_R_body('base')
        }
        return result_dict

    def pd_step(self,trgt=None,ctrl_idxs=None,nstep=1,INCREASE_TICK=True):
        """
            Step with PD Controller
        """

        qpos = self.get_q(self.ctrl_joint_idxs)
        self.PID.update(x_trgt=trgt,t_curr=self.get_sim_time(),x_curr=qpos,VERBOSE=False)
        torque = self.PID.out()

        super().step(ctrl=torque,ctrl_idxs=ctrl_idxs,nstep=nstep,INCREASE_TICK=INCREASE_TICK)

        # actor_root_states = position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

        result_dict = {
            "actor_root_states" : np.concatenate((self.get_p_body('base'), r2quat(self.get_R_body('base'))[[1,2,3,0]], self.get_qvel_joint('base')[0:3], self.get_qvel_joint('base')[3:6]), axis=-1),
            "dof_pos": self.get_qposes(),
            "dof_vel": self.get_qvels(),
            "rigid_body_pos": self.get_ps(),
            "contact_info": self.get_contact_info()
            # "root_p": self.get_p_body('base'),
            # "root_R": self.get_R_body('base')
        }
        return result_dict


    # def pd_step(self, trgt, PID):
    #     """
    #         Step
    #     """
    #     qpos = self.get_q(self.ctrl_joint_idxs)
    #     #.data.qpos[env.ctrl_joint_idxs] # joint position
    #     # qvel = env.data.qvel[env.ctrl_qvel_idxs] # joint velocity
    #     PID.update(x_trgt=trgt,t_curr=self.get_sim_time,x_curr=qpos,VERBOSE=True)
    #     torque = PID.out()

    #     self.step(ctrl=torque)
    #     print('hey')



    
    def render_queue(self, queue, render_every=1):
        """
            Render
        """
        if self.is_running:
            return
        r=queue.get()
        print("queue value: ", r)

        self.is_running = True
        self.init_viewer()

        self.render(render_every=render_every)

        time.sleep(10*r)
        print("test end")
        self.is_running = False
        self.viewer.close()
