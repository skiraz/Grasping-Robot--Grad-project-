import os
import cv2
import sys
import argparse
import stable_baselines3
from stable_baselines3 import PPO as PPO2
from stable_baselines3.common.noise  import OrnsteinUhlenbeckActionNoise
from qibullet.camera import CameraRgb
from qibullet.camera import CameraDepth
from datetime import datetime
import math
from stable_baselines3.common.logger import configure



from qibullet import Camera
import time
import numpy as np
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG
from stable_baselines3 import A2C
from stable_baselines3 import SAC

import gymnasium
from gymnasium import spaces
import threading

import pybullet as p
import pybullet_data
from qibullet import PepperVirtual
from qibullet import SimulationManager
from urllib.request import Request, urlopen



class PepperEnv(gymnasium.Env):    
    def __init__(self,gui = True):

        # the kinematic chain which will be used to create the policy
        self.Left_kinematic_chain = [
            "LShoulderRoll",
            # "LShoulderPitch",
            "LElbowRoll"]
            # "LElbowYaw",
            
        
        # list of values defining the robot initial state

        self.initial_stand = [
            1.562,
            # -0.439,
            -0.048,
            # 0.154,
            ]

        self.joints_initial_pose = list()

        # variables defining the rl episodes
        self.episode_start_time = None
        self.episode_over = False
        self.episode_failed = True
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0

        self.simulation_manager = SimulationManager()

        self.gui = gui

        self._setupScene()
        time.sleep(0.1)



        lower_limits = list()
        upper_limits = list()


        # kinematics ranges
        lower_limits.extend([self.pepper.joint_dict[joint].getLowerLimit() for
                            joint in self.Left_kinematic_chain])
        upper_limits.extend([self.pepper.joint_dict[joint].getUpperLimit() for
                            joint in self.Left_kinematic_chain])
        

        
        # Add gripper position and rotation to the limits
        lower_limits.extend([-2, -2, 0, -1, -1, -1, -1])
        upper_limits.extend([2, 2, 3, 1, 1, 1, 1])
        
        
        

        # self.observation_space = gym.spaces.Dict({
        #     "kinematics":gym.spaces.Box(low = np.array(lower_limits),
        #                                 high = np.array(upper_limits),shape=(12,)),
                                        
                                        
        #     "image":gym.spaces.Box(low=0, high=255, shape=(320, 240, 3)) 
        #     })
        

        
        # self.kinematics_obs = gym.spaces.Box(low = np.array(lower_limits),
        #                                 high = np.array(upper_limits))
        

        # self.image_obs_space =gym.spaces.Box(low=0, high=255, shape=(320, 240, 3))


        # self.observation_space = gym.spaces.Tuple([self.kinematics_obs, 
        #                                 self.image_obs_space])
        

        self.observation_space = spaces.Dict({
            "kinematics" : spaces.Box(
                                      low = np.array(lower_limits),          
                                      high = np.array(upper_limits)),
            "image" :  spaces.Box(low=0, high=255, shape=(240, 320, 3))
                })

        # x_bounds = (-10, 10)  # Minimum and maximum values for the x-coordinate
        # y_bounds = (-10, 10)  # Minimum and maximum values for the y-coordinate
        # z_bounds = (-10, 10)  # Minimum and maximum values for the z-coordinate


        # soda_can = spaces.Box(low=np.array([x_bounds[0], y_bounds[0], z_bounds[0]]),
        #                  high=np.array([x_bounds[1], y_bounds[1], z_bounds[1]]),
        #                  dtype=np.float32)
        
        # self.observation_space = spaces.Dict({
        #     "kinematics" : spaces.Box(
        #                               low = np.array(lower_limits),          
        #                               high = np.array(upper_limits)),
        #     "soda_can" :  soda_can
        #         })
        

    


        

        velocity_limits = [
            self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.Left_kinematic_chain]
        velocity_limits.extend([
            -self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.Left_kinematic_chain])
        

        normalized_limits = self.normalize(velocity_limits)
        self.max_velocities = normalized_limits[:len(self.Left_kinematic_chain)]
        self.min_velocities = normalized_limits[len(self.Left_kinematic_chain):]



        self.action_space = spaces.Box(
            low=np.array(self.min_velocities),
            high=np.array(self.max_velocities)
            )

        

        self.robot_body_id = self.pepper.robot_model

        self.   models_dir = "models"
        self.logdir = "logs"


        self.make_dirs()



        '''depth image'''
        thread = threading.Thread(target=self.camera_view_depth,args=(self.pepper,))
        thread.start()

        

        # self.camera_thread = threading.Thread(target=self.camera_view,args=(self.pepper,))

        # # Auto stepping set to False, the user has to manually step the simulation
        # self.client = self.simulation_manager.launchSimulation(gui=self.gui, auto_step=True)
        # self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)
        # self.robot_body_id = self.pepper.robot_model
        # self.camera_thread = threading.Thread(target=self.camera_view,args=(self.pepper,))
        # self.joints_thread = threading.Thread(target=self.joints,args=(self.pepper,))
        # setting pepper posture 
        

        # self.pepper.goToPosture("Stand",percentage_speed=1)
        # # delay for enviroment preperation =
        # time.sleep(3)





        
        
        # self.joint_parameters = list()

        # for name, joint in self.pepper.joint_dict.items():
        #     if "Finger" not in name and "Thumb" not in name:
        #         self.joint_parameters.append((
        #             p.addUserDebugParameter(
        #                 name,
        #                 joint.getLowerLimit(),
        #                 joint.getUpperLimit(),
        #                 self.pepper.getAnglesPosition(name)),
        #             name))
                
                
    def normalize(self, values, range_min=-1.0, range_max=1.0):
        """
        Normalizes values (list) according to a specific range
        """
        zero_bound = [x - min(values) for x in values]
        range_bound = [
            x * (range_max - range_min) / (max(zero_bound) - min(zero_bound))
            for x in zero_bound]
        
        return [x - max(range_bound) + range_max for x in range_bound]

    def make_dirs(self):
        if not os.path.exists(self.models_dir):
            os.mkdir(self.models_dir)

        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)


    # TODO: make model.save func and load func 
    
    
    def _setupScene(self):
        # print(342432423)
        """
        Setup a scene environment within the simulation
        """
        self.client = self.simulation_manager.launchSimulation(gui=self.gui)
        self.pepper = self.simulation_manager.spawnPepper(
            self.client,
            spawn_ground_plane=True)
        
        

        self.pepper.goToPosture("Stand", 1)
        initial_stand = [
            -1.562,
            -0.439,
            -0.048,
            0.154,
            ]
        Left_kinematic_chain = [
            "LShoulderRoll",
            "LShoulderPitch",
            "LElbowRoll",
            "LElbowYaw",
            ]

        self.pepper.setAngles(  
            Left_kinematic_chain,
            initial_stand,
            1.0)
        # self.pepper.setAngles(["LShoulderRoll"],[0.009],1)
        self.pepper.setAngles(["LShoulderRoll","HeadPitch","HeadYaw"],[1.562, 0.45,0.198],1)
        self.pepper.setAngles(["LHand"],[0.98],1)
        self.pepper.setAngles(["LWristYaw"],[-0.015],1)
        time.sleep(1)


       
        

            


        self.joints_initial_pose = self.pepper.getAnglesPosition(
            self.pepper.joint_dict.keys())
        
        # spawining table 

        table_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.4, 0.4, 0.02],
            rgbaColor=[0.9, 0.9, 0.9, 1.0])
        
        table_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.4, 0.4, 0.02])
        
        self.table_position = [0.7, 0.3, .9  ]  # Modify this position to place the table where you want
        
        self.table_body_id = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0.0, 0.0, 0.0],
            baseCollisionShapeIndex=table_collision_shape_id,
            baseVisualShapeIndex=table_visual_shape_id,
            basePosition=self.table_position)



        # spawning can
        self.soda = 1
        self.water = 0
        self.trash_metal = 1
        self.trash_paper = 1
        if self.soda :
            self.soda_pos = [0.35, 0.15, .96]
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.soda = p.loadURDF(
                r"D:\projects\graduation project\qi bullet simulation\soda_can.urdf",
                basePosition=self.soda_pos,
                globalScaling=1,
                physicsClientId=self.client)
            
   


        if self.water :
            self.water_bottle_pos = [0.345, 0.1, .9]
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.water_bottle = p.loadURDF(
                r"D:\projects\graduation project\qi bullet simulation\water_bottle.urdf",
                basePosition=self.water_bottle_pos,
                globalScaling=1,
                physicsClientId=self.client)
        
        if self.trash_metal:
            self.trash_Can_metal_pos = [-0.4, -0.4, .0]
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.trash_Can = p.loadURDF(
                    r"D:\projects\graduation project\qi bullet simulation\qi-gym\data\trash_metal.urdf",
                    basePosition=self.trash_Can_metal_pos,
                    globalScaling=2,
                    physicsClientId=self.client)
            

        if self.trash_paper:
            self.trash_Can_paper_pos= [-0.4, 0.4, .0]
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.trash_Can = p.loadURDF(
                    r"D:\projects\graduation project\qi bullet simulation\qi-gym\data\trash_papper.urdf",
                    basePosition=self.trash_Can_paper_pos,
                    globalScaling=2,
                    physicsClientId=self.client)
        
        
        time.sleep(0.3)





    def step(self, action):
        # print(11111111111111111111111)
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        try:
            action = list(action)
            assert len(action) == len(self.action_space.high.tolist())

        except AssertionError:
            print("Incorrect action")
            return None, None, None, None

        self.episode_steps += 1
        np.clip(action, self.min_velocities, self.max_velocities)
        self._setVelocities(self.Left_kinematic_chain, action)

        obs, reward = self._getState()
        # print(obs,reward,self.episode_steps)
        return obs, reward, self.episode_over, 0,{}


    
        
    def _getSodaPosition(self):
        """
        Returns the position of the target soda in the world
        """
        # Get the position of the soda (goal) in the world
        soda_pose, soda_qrot = p.getBasePositionAndOrientation(
            self.soda)

        return soda_pose, soda_qrot
    


    def _getState(self):
        """
        Gets the observation and computes the current reward. Will also
        determine if the episode is over or not, by filling the episode_over
        boolean. When the euclidian distance between the wrist link and the
        cube is inferior to the one defined by the convergence criteria, the
        episode is stopped
        """
        reward = 0.0

        # Get position of the object and gripper pose in the odom frame
        # projectile_pose, _ = self._getProjectilePosition()
        # bucket_pose, _ = self._getBucketPosition()

        # Check if there is a self collision on the r_wrist, if so stop the
        # episode. Else, check if the ball is below 0.049 (height of the
        # trash's floor)
        # if self.pepper.isSelfColliding("l_wrist") or\
       #     self.episode_over = True
            # self.episode_failed = True

       
        


        
            
        # elif projectile_pose[2] <= 0.049:
        #     self.episode_over = True
        if (time.time() - self.episode_start_time) > 8:
            reward += -10
            self.episode_over = True
            self.episode_failed = True

        


        

        # self.table_coll = False
        # self.soda_coll = False
        # self.text = "no"
        # for link_id in list(range(32,50,1)):
        #             # Check if there are any contact points between the link and the can
        #             contact_points = p.getContactPoints(self.robot_body_id, self.soda, linkIndexA=link_id)
        #             if len(contact_points) > 0:
        #                 # print(322222222222222)
        #                 self.text = "Collision detected between the can and link : "+str(link_id)
        #                 reward += +10
        #                 self.episode_failed = False
        #                 self.episode_over = True
        #                 self.soda_coll= True
        #                 # self.soda_coll = True
        # # if not self.soda_coll:
        # #     reward += -10

        

        

        
        
        
                        

        # for link_id in range(p.getNumJoints(self.robot_body_id)):
        #             # Check if there are any contact points between the link and the table
        #             contact_points = p.getContactPoints(self.robot_body_id, self.table_body_id, linkIndexA=link_id)
        #             if len(contact_points) > 0:
        #                 # # print("Collision detected between the table and link", link_id)
        #                 # reward += -100
        #                 # self.episode_over = True
        #                 # self.episode_failed = True
        #                 self.table_coll=True
        #                 reward += -3
        #                 self.episode_over= True
                        
        # SodaPos,_ = self._getSodaPosition()
        # reward += - np.sqrt((SodaPos[0]-linkPos[0])**2+(SodaPos[1]-linkPos[1])**2+(SodaPos[2]-linkPos[2])**2)*3
        # self.episode_reward += reward


        



        
                        

        # Fill the observation
        
       

        # top_link_index = 1  # index of the "black" link
        # top_link_state = p.getLinkState(self.soda, top_link_index)
        # top_link_pos, top_link_orient = top_link_state[0], top_link_state[1]

        # SodaPos = (top_link_pos[0], top_link_pos[1], top_link_pos[2] + 0.0518/2)


        
        
        # return distance 

       

        # reward += -distance

        # if self.table_coll:
            

        
        

        SodaPos,_ = self._getSodaPosition()
        linkPos ,_ = self._getLinkPosition("l_gripper")
        distance = np.sqrt(np.sqrt( (self.soda_pos[0]-linkPos[0])**2+(self.soda_pos[1]-linkPos[1])**2+(self.soda_pos[2]+0.15-linkPos[2])**2 ))
        reward += 100 / distance+1
        
        self.episode_reward += reward
        self.episode_number += 1


        
        # self.episode_reward += reward
        if distance <=0.28:
            self.episode_over = True
            self.episode_failed = False

        
        if self.episode_over:
            self._printEpisodeSummary()
        obs = self._getObservation()
        

        return obs, reward
        #         self.pepper.isSelfColliding("LForeArm"):
        #     reward += -50
     



    def _hardResetJointState(self):

        """
        Performs a hard reset on the joints of the robot, avoiding the robot to
        get stuck in a position
        """
        for joint, position in\
                zip(self.pepper.joint_dict.keys(), self.joints_initial_pose):
            p.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[joint].getIndex(),
                p.VELOCITY_CONTROL,
                targetVelocity=0.0)
            p.resetJointState(
                    self.pepper.robot_model,
                    self.pepper.joint_dict[joint].getIndex(),
                    position)
    

    def _resetScene(self):
        """
        Resets the scene for a new scenario
        """
        p.resetBasePositionAndOrientation(
                self.pepper.robot_model,
            posObj=[0.0, 0.0, 0.0],
            ornObj=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client)

        self._hardResetJointState()
        
        # return (self._getObservation(),info)
        import random

        random_num = random.uniform(0.03, 0.28)

        self.soda_pos = [0.37, random_num, .96]
        

      
        if self.soda:
            p.resetBasePositionAndOrientation(
                self.soda,
                posObj=self.soda_pos,    
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client)
        if self.water:
            p.resetBasePositionAndOrientation(
                self.table_body_id,
                posObj=self.table_position,
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client)

        # time.sleep(0.4)

        

    def _printEpisodeSummary(self, info_dict={}):
        """
        Prints a summary for an episode

        Parameters:
            info_dict - Dictionnary containing extra data to be displayed
        """
        if self.episode_failed:
            episode_status = "FAILURE"
        else:
            episode_status = "SUCCESS"

        print("#---------Episode-Summary---------#")
        print("Episode number: " + str(self.episode_number))
        print("Episode's number of steps: " + str(self.episode_steps))
        print("Episode status: " + episode_status)
        print("Episode reward: " + str(self.episode_reward))
        # print("Episode soda_coll: " , self.soda_coll)
        # print("Episode soda_coll Text: " , self.text)
        # print("Episode table_coll: " , self.table_coll)
        print("Episode soda pos: " , self.soda_pos)

        for key, value in info_dict.items():
            print(key + ": " + str(value))

    def _termination(self):
        """
        Terminates the environment
        """
        self.simulation_manager.stopSimulation(self.client)


    def _getLinkPosition(self, link_name):
        """
        Returns the position of the link in the world frame
        """
        link_state = p.getLinkState(
            self.pepper.robot_model,
            self.pepper.link_dict[link_name].getIndex())
        
        return link_state[0], link_state[1]



    def _getObservation(self):
        """
        Returns the observation

        Returns:
            obs - the list containing the observations
        """
        # Get position of the projectile and bucket in the odom frame
        # projectile_pose, _ = self._getProjectilePosition()
        # bucket_pose, _ = self._getBucketPosition()


        # Get the position of the r_gripper in the odom frame (base footprint
        # is on the origin of the odom frame in the xp)
        gripper_pose, gripper_rot = self._getLinkPosition("l_gripper")

        SodaPos,_ = self._getSodaPosition()
        # linkPos ,_ = self._getLinkPosition("l_gripper")
        # dist = np.sqrt((SodaPos[0]-linkPos[0])**2+(SodaPos[1]-linkPos[1])**2+(SodaPos[2]-linkPos[2])**2)**3
        SodaPos = [SodaPos[0],SodaPos[1],SodaPos[2]]
        

        image = self.camera_view(self.pepper)
        # # Fill and return the observation
        
        kinematics = self.pepper.getAnglesPosition(self.Left_kinematic_chain) +[pose for pose in gripper_pose] +[rot for rot in gripper_rot]


        obs = {"kinematics":kinematics,"image":image}
        # print(self.depth_image)
        return  obs



    def _setVelocities(self, angles, normalized_velocities):
        """
        Sets velocities on the robot joints
        """
        for angle, velocity in zip(angles, normalized_velocities):
            # Unnormalize the velocity
            velocity *= self.pepper.joint_dict[angle].getMaxVelocity()

            position = self.pepper.getAnglesPosition(angle)
            lower_limit = self.pepper.joint_dict[angle].getLowerLimit()
            upper_limit = self.pepper.joint_dict[angle].getUpperLimit()

            if position <= lower_limit and velocity < 0.0:
                velocity = 0.0
                self.episode_failed = True
            elif position >= upper_limit and velocity > 0.0:
                velocity = 0.0
                self.episode_failed = True

            p.setJointMotorControl2(
                self.pepper.robot_model,
                self.pepper.joint_dict[angle].getIndex(),
                p.VELOCITY_CONTROL,
                targetVelocity=velocity/2,
                force=self.pepper.joint_dict[angle].getMaxEffort())

    def camera_view(self,perspective):
        
            # handle_top = robot.subscribeCamera(
            # PepperVirtual.ID_CAMERA_TOP,
            # resolution=Camera.K_QVGA,
            # fps=40)

        # Same process for the bottom camera
            handle_bottom = perspective.subscribeCamera(
            PepperVirtual.ID_CAMERA_BOTTOM,
            resolution=Camera.K_QVGA,
            fps=20)

            # img_top = perspective.getCameraFrame(handle_top)
            img_bottom = perspective.getCameraFrame(handle_bottom)
            # if not taken:
            #     cv2.imwrite("soda.jpg",img_bottom)
            #     taken = True

            # cv2.imshow("top camera", img_top)
            # print(img_bottom.shape)
            return img_bottom
    


    def camera_view_depth(self,robot):
        taken = False
        while True:
            handle_depth = robot.subscribeCamera(
                PepperVirtual.ID_CAMERA_DEPTH,
                resolution=Camera.K_QVGA,
                fps=20)

            depth_image = robot.getCameraFrame(handle_depth)
            # depth_image = np.reshape(img_depth[3], (img_depth[1], img_depth[0]))

            # Normalize the depth values
            # normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # Apply a color map to the depth image
            # depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
            depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # Invert the depth values (so closer objects are lighter)
            depth_image_inverted = 255 - depth_image_normalized

            # Apply a square root mapping to the depth values
            depth_image_sqrt = np.sqrt(depth_image_inverted)

            # Normalize the mapped depth image to 0-255 range
            depth_image_normalized = cv2.normalize(depth_image_sqrt, None, 0, 255, cv2.NORM_MINMAX)

            # Convert the depth image to 8-bit unsigned integer format
            depth_image_uint8 = depth_image_normalized.astype(np.uint8)

            # Apply a color map to the depth image
            depth_colormap = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)
            self.depth_image = depth_colormap


            # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # # Invert the depth values (so closer objects are lighter)
            # depth_image_inverted = 255 - depth_image_normalized

            # # Apply a color map to the depth image
            # depth_colormap = cv2.applyColorMap(depth_image_inverted.astype(np.uint8), cv2.COLORMAP_JET)


            cv2.imshow("Depth Camera", depth_colormap)
            cv2.imwrite("depth image simulation.jpg", depth_colormap)
            cv2.waitKey(1)
    


    def reset(self):
        """
        Resets the environment for a new episode
        """
        self.episode_over = False
        self.episode_failed = True
        self.episode_reward = 0.0
        self.episode_steps = 0
        self._resetScene()
        time.sleep(0.01)

        # Reset the start time for the current episode
        self.episode_start_time = time.time()

        # Fill and return the observation
        # time.sleep(1)
        return self._getObservation(),""
    

    # def reset(self):
    #     # self.simulation_manager.resetSimulation(self.client)
    #     self.pepper.goToPosture("stand",percentage_speed=1)
    #     # print(self.pepper.getAnglesPosition("LShoulderPitch"))
    #     p.removeAllUserDebugItems()
    #     self.joint_parameters = list()

    #     for name, joint in self.pepper.joint_dict.items():
    #         if "Finger" not in name and "Thumb" not in name:
    #             self.joint_parameters.append((
    #                 p.addUserDebugParameter(
    #                     name,
    #                     joint.getLowerLimit(),
    #                     joint.getUpperLimit(),
    #                     self.pepper.getAnglesPosition(name)),
    #                 name))
    #     time.sleep(2)
                
                  
            
    

   



    def train_model(self,model_name,save_dir,timesteps,lr=0.01,log_dir="logs",):
        if model_name == "A2C":
            model = A2C(  
            "MultiInputPolicy",
            env,
            gamma=0.99,
            verbose=1,
            learning_rate=lr,
            tensorboard_log = "logs"
            )


            #Training the model
            model.learn(total_timesteps=timesteps,tb_log_name=save_dir)
            

            # saving the model
            model.save(os.path.join(env.models_dir,save_dir))
            print("END OF LEARNING")


        if model_name == "PPO":
            model = PPO2(  
            "MultiInputPolicy",
            env,
            gamma=0.99,
            verbose=1,
            learning_rate=lr,
            tensorboard_log = "logs"
            )

             #Training the model
            model.learn(total_timesteps=timesteps,tb_log_name=save_dir)
            

            # saving the model
            model.save(os.path.join(env.models_dir,save_dir))
            print("END OF LEARNING")


        if model_name == "DDPG":
            model = DDPG(  
            "MultiInputPolicy",
            env,
            gamma=0.99,
            verbose=1,
            learning_rate=lr,
            tensorboard_log = "logs"
            )


            
            #Training the model
            model.learn(total_timesteps=timesteps,tb_log_name=save_dir)
            

            # saving the model
            model.save(os.path.join(env.models_dir,save_dir))
            print("END OF LEARNING")







        

        
    
    def launch_simulation(self):
        
         
        self.camera_thread.start()
        print(self.joint_parameters)
        # self.joints(self.pepper)
        try:
            while True:
                print(self.pepper.getAnglesPosition("LShoulderPitch"))
            
                for joint_parameter in self.joint_parameters:
                    self.pepper.setAngles(
                        joint_parameter[1],
                        p.readUserDebugParameter(joint_parameter[0]), 1)
                    # table_position = p.getBasePositionAndOrientation(self.table_body_id)[0]
                    # soda_can_position = [table_position[0], table_position]
                


                # Loop over all links in the robot
                for link_id in [48,49]:
                    # Check if there are any contact points between the link and the ball
                    contact_points = p.getContactPoints(self.robot_body_id, self.soda, linkIndexA=link_id)
                    if len(contact_points) > 0:
                        print("Collision detected between the ball and link", link_id)
                        # self.reset()

                for link_id in range(p.getNumJoints(self.robot_body_id)):
                    # Check if there are any contact points between the link and the ball
                    contact_points = p.getContactPoints(self.robot_body_id, self.table_body_id, linkIndexA=link_id)
                    if len(contact_points) > 0:
                        print("Collision detected between the table and link", link_id)
                        # self.reset()
                # camera_view(robot)
                
            

                


                # Step the simulation
                self.simulation_manager.stepSimulation(self.client)

        except KeyboardInterrupt:
            pass
        finally:
            self.simulation_manager.stopSimulation(self.client)




def main():
     pass
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--generate_pretrain",
    #     type=int,
    #     default=0,
    #     help="If true, launch an interface to generate an expert trajectory")

    # parser.add_argument(
    #     "--train",
    #     type=int,
    #     default=1,
    #     help="True: training, False: using a trained model")

    # parser.add_argument(
    #     "--algo",
    #     type=str,
    #     default="ppo2",
    #     help="The learning algorithm to be used (ppo2 or ddpg)")

    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     default="",
    #     help="The version name of the model")

    # parser.add_argument(
    #     "--gui",
    #     type=int,
    #     default=1,
    #     help="Wether the GUI of the simulation should be used or not. 0 or 1")

    # args = parser.parse_args()
    # algo = args.algo.lower()

    # try:
    #     assert args.gui == 0 or args.gui == 1
    #     assert algo == "ppo2" or algo == "ddpg"

    # except AssertionError as e:
    #     print(str(e))
    #     return

    # env = PepperEnv(gui=args.gui)
    # vec_env = DummyVecEnv([lambda: env])

    # # Generate an expert trajectory
    # if args.generate_pretrain:
    #     pass
    
    # # Train a model
    # elif args.train == 1:
    #     pass
    #     # while True:
    #     #     # req = Request(
    #     #     #     "https://frightanic.com/goodies_content/docker-names.php",
    #     #     #     headers={'User-Agent': 'Mozilla/5.0'})

    #     #     # webpage = str(urlopen(req).read())
    #     #     # word = webpage.split("b\'")[1]
    #     #     # word = word.split("\\")[0]
    #     #     # word.replace(" ", "_")

    #     #     # try:
    #     #     #     assert os.path.isfile(
    #     #     #         "models/" + algo + "_throw_" + word + ".pkl")

    #     #     # except AssertionError:
    #     #     #     break

    #     # log_name = "./logs/throw/" + word

    #     # if algo == "ppo2":
    #     #     # For recurrent policies, nminibatches should be a multiple of the 
    #     #     # nb of env used in parallel (so for LSTM, 1)
    #     #     model = PPO2(
    #     #         MultiInputPolicy,
    #     #         vec_env,
    #     #         # nminibatches=1,
    #     #         verbose=0,
    #     #         tensorboard_log=log_name)

    #     # elif algo == "ddpg":
    #     #     action_noise = OrnsteinUhlenbeckActionNoise(
    #     #         mean=np.zeros(env.action_space.shape[-1]),
    #     #         sigma=float(0.5) * np.ones(env.action_space.shape[-1]))

    #     #     model = DDPG(
    #     #         "MlpPolicy",
    #     #         env,
    #     #         verbose=0,
    #     #         param_noise=None,
    #     #         action_noise=action_noise,
    #     #         tensorboard_log=log_name)

    #     # try:
    #     #     model.learn(total_timesteps=1000000)
        
    #     # except KeyboardInterrupt:
    #     #     print("#---------------------------------#")
    #     #     print("Training \'" + word + "\' interrupted")
    #     #     print("#---------------------------------#")
    #     #     sys.exit(1)


    #     # model.save("models/" + algo + "_throw_" + word)

    # # Use a trained model
    # else:
    #     if args.model == "":
    #         print("Specify the version of the model using --model")
    #         return

    #     if algo == "ppo2":
    #         model = PPO2.load("models/" + algo + "_throw_" + args.model)
    #     elif algo == "ddpg":
    #         model = DDPG.load("models/" + algo + "_throw_" + args.model)

    #     for test in range(10):
    #         dones = False
    #         obs = env.reset()

    #         while not dones:
    #             action, _states = model.predict(obs)
    #             obs, rewards, dones, info = env.step(action)

    # time.sleep(2)
    # env._termination()

  

if __name__ == "__main__":
    tmp_path = "/logs/iter 15"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    env = PepperEnv(gui=1)

    vec_env = DummyVecEnv([lambda: env])

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[-1]),sigma=float(0.5) * np.ones(env.action_space.shape[-1]))


    # model = PPO2.load(os.path.join(env.models_dir,"iter 14"),env=env)
    # model.learning_rate=0.001
    # model.batch_size=128
    # # model = PPO2(  
    # #     "MultiInputPolicy",
    # #     env,
    # #     # buffer_size=100,
    # #     gamma=0.99,
    # #     verbose=1,
         
    # #     learning_rate=0.001
    # #     # action_noise=action_noise
    # #     )
    # #             # buffer_size=100)
    

    # model.learn(total_timesteps=10 ,tb_log_name="/logs/iter 15")
     


    
    
    # model.save(os.path.join(env.models_dir,"iter 15"))
    # print("END OF LEARNING")
    # # model = A2C.load(r"D:\projects\graduation project\qi bullet simulation\models\iter 1.zip",env=env)
    for i in [0.0001]:

        model = PPO2(  
            "MultiInputPolicy",
            env,
            # buffer_size=100,
            gamma=0.99,
            verbose=1,
            learning_rate=i,
            tensorboard_log = r"logs/PPO_IMAGE"
            # action_noise=action_noise
            )
                    # buffer_size=100)
        # tensorboard_callback = TensorBoardCallback(log_dir="/logs/iter 15")

        
        # model.set_logger(new_logger)
        model.learn(total_timesteps=10000,tb_log_name=f"ppo_IMAGE_lr_{i}")


        
        model.save(os.path.join(env.models_dir,f"ppo_IMAGE_lr_{i}"))
        print("END OF LEARNING")

        

    # obs = env.reset()``   ``
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()


    








