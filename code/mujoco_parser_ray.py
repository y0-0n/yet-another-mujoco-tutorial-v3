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
from util import sample_xyzs,rpy2r,r2quat
import matplotlib.pyplot as plt

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
                k_p = 0.4, k_i = 0.01, k_d = 0.001,
                out_min = self.ctrl_ranges[:,0],
                out_max = self.ctrl_ranges[:,1],
                dt = 0.02,
                ANTIWU  = True)
        self.env_id = env_id
        # Change floor friction
        self.model.geom('floor').friction = np.array([1,0.01,0]) # default: np.array([1,0.01,0])
        self.model.geom('floor').priority = 1 # >0
        if (self.VERBOSE):
            print("PID controller ready")
            print('Floor friction: ',self.model.geom('floor').friction)
            print('Floor priority: ',self.model.geom('floor').priority)
    
    def play_steps(self):
        raise NotImplementedError("play_steps not implemented")
        #for n in range(self.horizon_length):
        # for n in range(16):
        #     self._

    # ______________________ AMP utils ______________________

    def reset_PID(self):
        self.PID.reset()
        if (self.VERBOSE):
            print("PID controller reset")

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

    def get_objects_poses(self):
        # get objects
        obj_names = [body_name for body_name in self.body_names
            if body_name is not None and (body_name.startswith("obj_"))]
        n_obj = len(obj_names)

        # Place objects
        colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])
        colors[:,3] = 1.0 # transparent objects
        obj_poses = np.empty((0, 7))
        for obj_idx,obj_name in enumerate(obj_names):

            geomadr = self.model.body(obj_name).geomadr[0]
            self.model.geom(geomadr).rgba = colors[obj_idx] # color

            jntadr  = self.model.body(obj_name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            
            obj_poses = np.append(obj_poses, np.expand_dims(self.data.qpos[qposadr:qposadr+7], axis=0), axis=0)

        return obj_poses

    def pd_step(self,trgt=None,ctrl_idxs=None,nstep=1,INCREASE_TICK=True, SAVE_VID=True):
        """
            Step with PD Controller
        """

        qpos = self.get_q(self.ctrl_joint_idxs)
        self.PID.update(x_trgt=trgt,t_curr=self.get_sim_time(),x_curr=qpos,VERBOSE=self.VERBOSE)
        torque = self.PID.out()

        super().step(ctrl=torque,ctrl_idxs=ctrl_idxs,nstep=nstep,INCREASE_TICK=INCREASE_TICK)

        # actor_root_states = position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

        # obj_poses = self.get_objects_poses()

        result_dict = {
            "actor_root_states" : np.concatenate((self.get_p_body('base'), r2quat(self.get_R_body('base'))[[1,2,3,0]], self.get_qvel_joint('base')[0:3], self.get_qvel_joint('base')[3:6]), axis=-1),
            "dof_pos": self.get_qposes(),
            "dof_vel": self.get_qvels(),
            "rigid_body_pos": self.get_ps(),
            "contact_info": self.get_contact_info(),
            # "obj_poses": obj_poses
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

    def throw_objects(self):
        # Throw cylinder
        obj_names = [body_name for body_name in self.body_names
            if body_name is not None and (body_name.startswith("obj_"))]
        n_obj = len(obj_names)

        # Place objects
        colors = np.array([plt.cm.gist_rainbow(x) for x in np.linspace(0,1,n_obj)])
        colors[:,3] = 1.0 # transparent objects
        for obj_idx,obj_name in enumerate(obj_names):
            xyzs = sample_xyzs(
                n_sample=n_obj,x_range=[0.45,1.65],y_range=[-0.38,0.38],z_range=[0.81,0.81],min_dist=0.2,xy_margin=0.05)

            geomadr = self.model.body(obj_name).geomadr[0]
            self.model.geom(geomadr).rgba = colors[obj_idx] # color

            jntadr  = self.model.body(obj_name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            qveladr = self.model.jnt_dofadr[jntadr]
            # geom_pos = self.data.qpos[qposadr:qposadr+3]
            # geom_pos[:2] = geom_pos[:2] + 0.005*np.random.randn(2)
            self.data.qpos[qposadr:qposadr+3] = self.data.qpos[0:3] + xyzs[obj_idx]
            self.data.qvel[qveladr:qveladr+3] = -xyzs[obj_idx]*5

            # self.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0,0,0])))



    
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
