MuJoCo AMP
==============================

The Adversarial Motion Prior (AMP) is a framework designed for imitation learning within physically simulated environments. It is variation of the [original AMP repository](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), which has been re-implemented specifically for the MuJoCo simulator. To make use of this code, you will need the MJCF file for your own rig, along with retargeted motion that is compatible with the specified rig.


Installation
==============================

After creating your environment, run the command:

    > pip install -r requirements.txt
##
Folder Structure
================

    .
    ├── asset
    │   ├── common_rig
    │   │   ├── common_rig_rpy.xml
    │   │   ├── common_rig.xml
    │   │   ├── motion                                      # CMU motion
    │   │   └── scene_common_rig.xml
    │   └── object
    │       ├── base_table.xml
    │       ├── floor_sky.xml
    │       ├── obj_cylinder.xml
    │       └── object_table.xml
    ├── code
    │   ├── amp                                             # AMP code
    │   ├── cfg                                             # AMP Configure
    │   │   ├── config.yaml
    │   │   ├── task
    │   │   └── train
    │   ├── demo_common_rig_01_ik.ipynb                     # Inverse Kknematics
    │   ├── demo_common_rig_02_kinematic_simulation.ipynb   # Kinematic simulation
    │   ├── main_amp.py                                     # main file
    │   ├── mujoco_parser.py                                # MuJoCo parser
    │   ├── mujoco_parser_ray.py                            # Extension of MuJoCo parser for parallel simulation
    │   ├── pid.py                                          # PID controller
    │   └── util.py
    ├── data
    │   └── VAAI_Non_M_01_de_01_results.pkl                 # NC motion
    ├── LICENSE
    ├── README.md

Run
============================================

To train AMP, run this command.

    > python main_amp.py
    
##

Configuration
==================================


In the cfg folder, you will find two important sub-folders:

### Task Configuration
The `task` folder contains configuration files specifying parameters for the task environment.

### Train Configuration
The `train` folder contains configuration files specifying parameters for the neural network training.

The configurations have been adapted from the [Original AMP code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), which means that certain parameters may not function as expected.

We have changed parameters such as 
- `num_envs` : number of environments for parallel simulation
- `horizon_length` : horizon length of episode in each simulator
- `headless` : if it's `true`, simulation will be rendered
- `train.params.config.minibatch_size` : It need to be changed depending on `num_envs` and `horizon_length`
- `checkpoint` : path of checkpoint
- `motion_file` : path of motion file (npy format)
- `assetFileName` : path of MJCF model (xml)


**Note** if you use the AMP: Adversarial Motion Priors environment in your work, please ensure you cite the following work:
```
@article{
	2021-TOG-AMP,
	author = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
	title = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2021},
	volume = {40},
	number = {4},
	month = jul,
	year = {2021},
	articleno = {1},
	numpages = {15},
	url = {http://doi.acm.org/10.1145/3450626.3459670},
	doi = {10.1145/3450626.3459670},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning},
} 
```