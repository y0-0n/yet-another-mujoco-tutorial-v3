import os
import torch

# from isaacgym import gymtorch
# from isaacgym import gymapi
# from isaacgym.torch_utils import *
import ray
from mujoco_parser_ray import MuJoCoParserClassRay
from mujoco_parser import MuJoCoParserClass
from amp.utils.torch_jit_utils import *
from ..base.vec_task import VecTask
from ray.util.queue import Queue
from amp.utils.constant import DOF_BODY_IDS, DOF_OFFSETS
# DOF_BODY_IDS    = [1, 2, 4, 5, 6,
#                    7, 8, 9, 10,
#                    12, 13, 14,
#                    16, 17, 18
#                     ]    # body idx which have DOF, not include root
# DOF_OFFSETS     = [0, 3, 4, 7, 8, 11,
#                    14, 15, 18, 21,
#                    24, 25, 28,
#                    31, 32, 35]  # joint number offset of each body
NUM_OBS = 1 + 6 + 3 + 3 + 65 + 35 + 12 # [(root_h(z-height):1, root_rot:6, root_vel:3, root_ang_vel:3, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 35    #from mjcf file (atlas_v5.xml actuator)
NUM_ACTORS_PER_ENVS = 1 # Added from JTM, 2 actors per envs

KEY_BODY_NAMES = ["right_hand", "left_hand", "right_ankle", "left_ankle"]

class CommonRigAMPBase(VecTask):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]
        self.num_balls = self.cfg["env"]["num_balls"]
        self.num_boxs = self.cfg["env"]["num_boxs"]
        self.is_soccer_task = self.cfg["env"]["is_soccer_task"]
        self.num_actors_per_envs = 1# + self.num_balls + self.num_boxs + (1 if self.is_soccer_task else 0) # Added from JTM, 2 actors per envs

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        self.queue = Queue(maxsize=4)
        
        # get gym GPU state tensors
        # actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        # dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim) # 공과 겹침
        # contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        actor_root_state = torch.zeros((self.num_envs, 13))
        
        # for env in self.mujoco_envs:
        #     ray.get(env.get_pR_body.remote('base'))
        #     actor_root_state[env.env_id, :3]


        #__________________________________________________________________________________________

        sensors_per_env = 2
        # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)

        # Added from JTM, Global state per environment
        # self._root_tensor = gymtorch.wrap_tensor(actor_root_state)
        self._root_states = torch.zeros((self.num_envs, 13), device=self.device)

        # Added from JTM, Set actor indices
        self.actor_indices = torch.arange(self.num_actors_per_envs * self.num_envs, dtype=torch.long, device=self.device).view(self.num_envs, self.num_actors_per_envs)
        self.humanoid_ids = self.actor_indices[:,0]
        # if self.is_soccer_task:
        #     self.soccer_ball_id = self.actor_indices[:,self.num_balls+self.num_boxs+1:].flatten()

        # self.ball_ids = self.actor_indices[:,1]
        # self.box_ids = self.actor_indices[:,2]

        # Added from JTM, Set actor states
        # self.humanoid_states = self._root_states[self.humanoid_ids]
        # self.ball_states = self._root_states[self.ball_ids]
        # self.box_states = self._root_states[self.box_ids]

        # Added from JTM, Initial root state
        # self._initial_root_states = self._root_states.clone()
        # self._initial_root_states[:, 7:13] = 0
        # self._ball_buffer = self._initial_root_states.clone()
        # self._box_buffer = self._initial_root_states.clone()

        # create some wrapper tensors for different slices
        self._dof_state = torch.zeros((self.num_envs * self.num_dof, 2), device=self.device)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        # right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "r_arm_shx")   # modified for Atlas
        # left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.envs[0], self.humanoid_handles[0], "l_arm_shx")    # modified for Atlas
        # self._initial_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        # self._initial_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        # self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        
        self._rigid_body_state = torch.zeros((self.num_envs * self.num_bodies, 13), device=self.device, dtype=torch.float)

        # Added from JTM, Set rigid body states
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:self.num_bodies-self.num_actors_per_envs+1, 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:self.num_bodies-self.num_actors_per_envs+1, 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:self.num_bodies-self.num_actors_per_envs+1, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:self.num_bodies-self.num_actors_per_envs+1, 10:13]

        self._contact_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        
        if self.viewer != None:
            self._init_camera()   
        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        # create MuJoCo envs
        self._create_ray_envs()

        # If randomizing, apply once immediately on startup before the fist sim step
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        # self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        raise NotImplementedError()
        # for env in self.mujoco_envs:
        #     pass
            
        # self.model.geom(geomadr).rgba = colors[obj_idx] # color

        # for i in range(self.num_envs):
        #     env_ptr = self.envs[i]
        #     handle = self.humanoid_handles[i]

        #     for j in range(self.num_bodies):
        #         self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
        #                                       gymapi.Vec3(col[0], col[1], col[2]))

        return

    def _create_ground_plane(self):
        raise NotImplementedError() # ground plane is created in the xml file
        # plane_params = gymapi.PlaneParams()
        # plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # plane_params.static_friction = self.plane_static_friction
        # plane_params.dynamic_friction = self.plane_dynamic_friction
        # plane_params.restitution = self.plane_restitution
        # self.gym.add_ground(self.sim, plane_params)
        return

    def _create_ray_envs(self):
        self.standard_env = MuJoCoParserClass(name='Common Rig',rel_xml_path=self.asset_file, VERBOSE=False,USE_MUJOCO_VIEWER=False)

        self._key_body_ids = self._build_key_body_ids_tensor() # NOTE: Different from Isaac index
        self._contact_body_ids = self._build_contact_body_ids_tensor() # NOTE: Different from Isaac index

        self.num_dof    = self.standard_env.n_rev_joint
        self.num_bodies = self.standard_env.n_body - 1 # Except worldbody
        self.num_joints = self.standard_env.n_rev_joint # NOTE: Different from Isaac num_joints

        motion_file = self.cfg['env'].get('motion_file')
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../asset/" + motion_file)
        self._load_motion(motion_file_path)
        # Ray parallelization
        self.mujoco_envs = [MuJoCoParserClassRay.remote(name='Common Rig Ray',
                                                        rel_xml_path=self.asset_file,
                                                        VERBOSE=False,
                                                        USE_MUJOCO_VIEWER=(not self.headless),
                                                        device=self.device,
                                                        env_id=i,
                                                        contact_body_ids=self._contact_body_ids.cpu(),
                                                        key_body_ids=self._key_body_ids.cpu(),
                                                        motion_lib=self._motion_lib,
                                                        max_episode_length=self.max_episode_length,
                                                        horizon_length=self.horizon_length,
                                                        ) for i in range(self.num_envs)]

        for i, env in enumerate(self.mujoco_envs):
            if self.headless == False:
                env.init_viewer.remote(viewer_title='Common Rig Ray'+str(i),viewer_width=1200,viewer_height=800,
                        viewer_hide_menus=True)
                env.update_viewer.remote(azimuth=174.08,distance=15,elevation=-23,lookat=[0.1,0.05,0.16],
                        VIS_TRANSPARENT=True,VIS_CONTACTPOINT=True,
                        contactwidth=0.2,contactheight=0.1,contactrgba=np.array([1,0,0,1]),
                        VIS_JOINT=True,jointlength=0.5,jointwidth=0.1,jointrgba=[0.2,0.6,0.8,0.6])


        self.right_foot_idx = self.standard_env.model.body('right_ankle').id # NOTE: Different from Isaac index
        self.left_foot_idx = self.standard_env.model.body('left_ankle').id # NOTE: Different from Isaac index

        self.torso_index = self.standard_env.model.body('torso').id # NOTE: Different from Isaac index

        # NOTE: No sensor
        # self.standard_env.model.sensor()


        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for j in range(self.num_dof):
            self.dof_limits_lower.append(self.standard_env.joint_ranges[j][0])
            self.dof_limits_upper.append(self.standard_env.joint_ranges[j][1])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        if (self._pd_control):
            self._build_pd_action_offset_scale()


        # self.PID.reset()

        return

        asset_options = gymapi.AssetOptions()
        # asset_options.fix_base_link = True
        asset_options.angular_damping = 0.05
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Added from JTM, Setup up ball & Create ball asset
        ball_radius = 0.11 # Soccer ball has ~ 11 cm radius.
        ball_options = gymapi.AssetOptions()
        ball_options.density = 100.27 # Soccer ball has 410~450g weight.
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_options)

        # # Added from JTM, Setup up box & Create box asset
        # box_size = 0.2
        # box_options = gymapi.AssetOptions()
        # box_options.density = 1000.0 # 1kg
        # box_asset = self.gym.create_box(self.sim, box_size, box_size, box_size, box_options)


        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_ankle") # modified for Atlas
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_ankle")   # modified for Atlas
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        # self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        # Added for JTM, ball asset add
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset) + (1 if self.is_soccer_task else 0)# + self.gym.get_asset_rigid_body_count(ball_asset) + self.gym.get_asset_rigid_body_count(box_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []

        # Added for JTM, object handles
        self.obj_handles = []

        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = -1
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "atlas", i, contact_filter, 0) # modified for Atlas

            # yoon0_0 collision
            pose = gymapi.Transform()
            pose.r = gymapi.Quat(0, 0, 0, 1)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            # Soccer ball
            if self.is_soccer_task:
                ball_pose = gymapi.Transform()
                ball_pose.p.x = 9
                ball_pose.p.y = 0
                ball_pose.p.z = 0.11
                soccer_ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "soccer_ball", i, contact_filter, 0)
                self.obj_handles.append(soccer_ball_handle)

            # Added for JTM, Set up ball, create ball asset
            ball_pose = gymapi.Transform()
            ball_pose.p.x = 3
            ball_pose.p.y = 0
            ball_pose.p.z = 0.6
            # ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_pose, "ball", i, contact_filter, 0)
            # self.obj_handles.append(ball_handle)

            # Added for JTM, Set up box, create box asset
            box_pose = gymapi.Transform()
            box_pose.p.x = 2
            box_pose.p.y = 0
            box_pose.p.z = 0.5
            # box_handle = self.gym.create_actor(env_ptr, box_asset, box_pose, "box", i, contact_filter, 0)
            # self.obj_handles.append(box_handle)


            if (self._pd_control):
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)
        
        if (self._pd_control):
            self._build_pd_action_offset_scale()

        return
    
    # def fetch_result(self):
    #     self._root_states = torch.zeros((self.num_envs, 13))
    #     self._dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)
    #     self._dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)
    #     self._rigid_body_state = torch.zeros((self.num_envs * self.num_bodies, 13), device=self.device, dtype=torch.float)
        
    # yoon0_0
    def _get_dof_axis_angle(self, dof_pos):
        num_joints = len(DOF_OFFSETS) - 1
        angle, axis = exp_map_to_angle_axis(dof_pos)
        axis_angle = torch.zeros((dof_pos.shape[0], num_joints*4))

        _1_dof_axis = [1, 2, 2, 1, 1] # spine, re, le, rk, lk
        _idx = 0
        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                angle, axis = exp_map_to_angle_axis(dof_pos[:, dof_offset:(dof_offset+dof_size)])
                axis_angle[:, 4*j:4*j+3] = axis# ZYX rotation order?
                axis_angle[:, 4*j+3] = angle
            elif (dof_size == 1):
                temp_dof_pos = torch.zeros((dof_pos.shape[0], 3))
                temp_dof_pos[:, _1_dof_axis[_idx]] = dof_pos[:, dof_offset]
                # axis_angle[:, 3*j:3*j+3] = temp_dof_pos # euler
                # axis_angle[:, 3*j:3*j+3] = euler_xyz_to_exp_map(axis_angle[:, 3*j], axis_angle[:, 3*j+1], axis_angle[:, 3*j+2]) #ZYX rotation order
                angle, axis = exp_map_to_angle_axis(temp_dof_pos)
                axis_angle[:, 4*j:4*j+3] = axis# ZYX rotation order?
                axis_angle[:, 4*j+3] = angle  
                _idx += 1

        return axis_angle

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1
        
        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if (dof_size == 3):
                lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

            elif (dof_size == 1):
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return

    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self._root_states, self.pre_root_states, self.humanoid_ids)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        # yoon0-0: compute obstacle reset

        return

    def _refresh_sim_tensors(self):
        raise NotImplementedError("_refresh_sim_tensors not used in mujoco envs")
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        if self.is_soccer_task:
            soccer_ball_obs = self._compute_soccer_ball_obs(env_ids)[:,:3]
            obs = torch.cat([obs, soccer_ball_obs],dim=1)

        if (env_ids is None):
            self.obs_buf[:] = obs
            # self.soccer_ball_obs_buf[:] = soccer_ball_obs
        else:
            
            self.obs_buf[env_ids] = obs
            # self.soccer_ball_obs_buf[env_ids] = soccer_ball_obs

        return
    
    def _compute_soccer_ball_obs(self, env_ids=None):
        if (env_ids is None):
            # root_states = self._root_states
            # Added from JTM, to fix the issue of the humanoid_ids not being defined
            root_states = self._root_states[self.soccer_ball_id]
        else:
            # root_states = self._root_states[env_ids]
            # Added from JTM, to fix the issue of the humanoid_ids not being defined
            root_states = self._root_states[self.soccer_ball_id[env_ids]]
        
        return root_states

    def _compute_humanoid_obs(self, env_ids=None):
        if (env_ids is None):
            # root_states = self._root_states
            # Added from JTM, to fix the issue of the humanoid_ids not being defined
            root_states = self._root_states[self.humanoid_ids]
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            # root_states = self._root_states[env_ids]
            # Added from JTM, to fix the issue of the humanoid_ids not being defined
            root_states = self._root_states[self.humanoid_ids[env_ids]]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            key_body_pos = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]

        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
                                        key_body_pos, self._local_root_obs)
            
        
        ## TODO: yoon0_0
        
        # obs = compute_humanoid_observations(root_states, dof_pos, dof_vel,
        #                                     key_body_pos, self._local_root_obs)
        # if env_ids is None:
        #     obs = torch.zeros((self.num_envs, NUM_OBS), device=self.device)
        # else:
        #     obs = torch.zeros((len(env_ids), NUM_OBS), device=self.device)
        return obs

    def _reset_actors(self, env_ids):
        raise NotImplementedError("_reset_actors not used in mujoco envs")
        # self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        # self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self._initial_root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 0
        # self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        self.pre_root_states = self._root_states.clone()

        # if (self._pd_control):
        #     # print(gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)))
        #     pd_tar = self._action_to_pd_targets(self.actions)
        #     pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)
        #     self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)
        # else:
        #     forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        #     force_tensor = gymtorch.unwrap_tensor(forces)
        #     self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

        return

    def post_physics_step(self):
        self.progress_buf += 1

        # self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.standard_env.model.body(body_name).id - 1 # worldbody is the first body
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.standard_env.model.body(body_name).id - 1 # worldbody is the first body
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        raise NotImplementedError()
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self._cam_prev_char_pos = self._root_states[0, 0:3].cpu().numpy()
        
        # cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], 
        #                       self._cam_prev_char_pos[1] - 3.0, 
        #                       1.0)
        # cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
        #                          self._cam_prev_char_pos[1],
        #                          1.0)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._root_states[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    dof_obs_size = 65
    dof_offsets = [0, 3, 4, 7, 8, 11,
                   14, 15, 18, 21,
                   24, 25, 28,
                   31, 32, 35]  # joint number offset of each body
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
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

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs

# yoon0-0 TODO: reward tuning
@torch.jit.script
def compute_humanoid_reward(cur_root_state, pre_root_state, humanoid_ids=None):
    # type: (Tensor, Tensor, Any) -> Tensor
    reward = torch.ones_like(cur_root_state[:, 0])
    return reward

@torch.jit.script
def compute_humanoid_reset2(progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                        max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros((1,))
    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[contact_body_ids, :] = 0 # masking contact body ids (foot, ...)
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1) # check body has fallen
        fall_contact = torch.any(fall_contact, dim=-1) # get fall env

        body_height = rigid_body_pos[..., 2]
        
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.ones((1,)) if has_fallen else terminated
    
    reset = torch.ones((1,)) if progress_buf >= max_episode_length - 1 else terminated

    return reset, terminated

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone() # (4096, 31, 3)
        masked_contact_buf[:, contact_body_ids, :] = 0 # masking contact body ids (foot, ...)
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1) # check body has fallen
        fall_contact = torch.any(fall_contact, dim=-1) # get fall env

        body_height = rigid_body_pos[..., 2]
        
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated