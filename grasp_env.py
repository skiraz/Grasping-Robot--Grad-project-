
import os
import gym
import time
import pybullet as p
import argparse
import stable_baselines3
from datetime import datetime
import numpy as np
from stable_baselines3.ppo.policies import CnnPolicy, MlpPolicy, MultiInputPolicy
from gym import spaces



# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.policies import MlpLstmPolicy

# from stable_baselines3.common.policies import MlpPolicy
# from stable_baselines3.common.policies import MlpLstmPolicy

# from stable_baselines3.ddpg.noise import AdaptiveParamNoiseSpec

# from stable_baselines3.ddpg.noise import AdaptiveParamNoiseSpec

# from stable_baselines3.ddpg.noise import AdaptiveParamNoiseSpec



from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO as PPO2
from stable_baselines3.common.noise  import OrnsteinUhlenbeckActionNoise

from stable_baselines3 import DDPG

import pybullet
import pybullet_data
from qibullet import PepperVirtual
from qibullet import SimulationManager
from urllib.request import Request, urlopen








class pepper_env(gym.Env):
      def __init__(self, gui=True):
        self.simulation_manager = SimulationManager()

        self.pepper_kinematic_chain = [
            # "KneePitch",
            # "HipPitch",
            # "HipRoll",
            "LShoulderPitch",
            "LShoulderRoll",
            "LElbowRoll",
            "LElbowYaw",
            "LWristYaw"]
        
        self.gui = gui
        client = self.simulation_manager.launchSimulation(gui=self.gui, auto_step=True)
        pepper = self.simulation_manager.spawnPepper(client, spawn_ground_plane=True)
        pepper_body_id = pepper.robot_model

        self.joints = [
            "RShoulderPitch",
            "RShoulderRoll",
            "RElbowRoll",
            "RElbowYaw",
            "RWristYaw"]

        # self.initial_stand = [
        #     1.207,
        #     -0.129,
        #     1.194,
        #     1.581,
        #     1.632]
        
        pepper.goToPosture("stand",percentage_speed=1)
        time.sleep(1)  

        # Passed to True at the end of an episode
        self.episode_start_time = None
        self.episode_over = False
        self.episode_failed = False
        self.episode_reward = 0.0
        self.episode_number = 0
        self.episode_steps = 0

        

        # # self.initial_bucket_pose = [0.65, -0.2, 0.0]
        # self.projectile_radius = 0.03

        self._setupScene()

        # lower_limits = list()
        # upper_limits = list()

        # # Bucket footprint in base_footprint and r_gripper 6D in base_footprint
        # # (in reality in odom, but the robot won't move and is on the odom
        # # frame): (x_b, y_b, x, y, z, rx, ry, rz)
        # # lower_limits.extend([-10, -10, -10, -10, 0, -7, -7, -7])
        # # upper_limits.extend([10, 10, 10, 10, 10, 7, 7, 7])
        # lower_limits.extend([-10, -10])
        # upper_limits.extend([10, 10])

        # # Add the joint positions to the limits
        # lower_limits.extend([self.pepper.joint_dict[joint].getLowerLimit() for
        #                     joint in self.r_kinematic_chain])
        # upper_limits.extend([self.pepper.joint_dict[joint].getUpperLimit() for
        #                     joint in self.r_kinematic_chain])

        # # Add gripper position to the limits
        # lower_limits.extend([-2, -2, 0, -1, -1, -1, -1])
        # upper_limits.extend([2, 2, 3, 1, 1, 1, 1])


        # self.observation_space = spaces.Box(
        #     low=np.array(lower_limits),
        #     high=np.array(upper_limits))

        # Define the action space
        velocity_limits = [
            self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain]
        velocity_limits.extend([
            -self.pepper.joint_dict[joint].getMaxVelocity() for
            joint in self.r_kinematic_chain])

        normalized_limits = self.normalize(velocity_limits)
        self.max_velocities = normalized_limits[:len(self.r_kinematic_chain)]
        self.min_velocities = normalized_limits[len(self.r_kinematic_chain):]

        self.action_space = spaces.Box(
            low=np.array(self.min_velocities),
            high=np.array(self.max_velocities))
        
        # setting up the table 
        table_visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.4, 0.4, 0.02],
            rgbaColor=[0.9, 0.9, 0.9, 1.0])

        

        table_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.4, 0.4, 0.02])
        
        
    
        self.table_position = [0.6, 0.0, .8  ]  # Modify this position to place the table where you want
    
        self.table_body_id = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0.0, 0.0, 0.0],
            baseCollisionShapeIndex=table_collision_shape_id,
            baseVisualShapeIndex=table_visual_shape_id,
            basePosition=self.table_position)
        
        # setting up the can 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ball = p.loadURDF(
            r"D:\projects\graduation project\qi bullet simulation\soda_can.urdf",
            basePosition=[0.345, 0.1, .9],
        
        globalScaling=1,
        physicsClientId=client)


        # self.observation_space = spaces.Box(
        #     low=np.array(lower_limits),
        #     high=np.array(upper_limits))



        def step(self, action):
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
            self._setVelocities(self.r_kinematic_chain, action)

            obs, reward = self._getState()
            return obs, reward, self.episode_over, {}
        
        def _getState(self, convergence_norm=0.15):

            """
            Gets the observation and computes the current reward. Will also
            determine if the episode is over or not, by filling the episode_over
            boolean. When the euclidian distance between the wrist link and the
            cube is inferior to the one defined by the convergence criteria, the
            episode is stopped

            """
            reward = 0.0

            # Get position of the object and gripper pose in the odom frame
            projectile_pose, _ = self._getProjectilePosition()
            bucket_pose, _ = self._getBucketPosition()

            # Check if there is a self collision on the r_wrist, if so stop the
            # episode. Else, check if the ball is below 0.049 (height of the
            # trash's floor)
            if self.pepper.isSelfColliding("r_wrist") or\
                    self.pepper.isSelfColliding("RForeArm"):
                self.episode_over = True
                self.episode_failed = True
                reward += -1
            elif projectile_pose[2] <= 0.049:
                self.episode_over = True
            elif (time.time() - self.episode_start_time) > 2.5:
                self.episode_over = True
                self.episode_failed = True
                reward += -1

            # Fill the observation
            obs = self._getObservation()

            # Compute the reward
            bucket_footprint = [bucket_pose[0], bucket_pose[1], 0.0]
            projectile_footprint = [projectile_pose[0], projectile_pose[1], 0.0]

            previous_footprint = [
                self.prev_projectile_pose[0],
                self.prev_projectile_pose[1],
                0.0]

            prev_to_target =\
                np.array(bucket_footprint) - np.array(previous_footprint)
            current_to_target =\
                np.array(bucket_footprint) - np.array(projectile_footprint)

            # test
            # init_to_current = np.array(projectile_footprint) - np.array([
            #     self.initial_projectile_pose[0],
            #     self.initial_projectile_pose[1],
            #     0.0])

            norm_to_target = np.linalg.norm(current_to_target)
            reward += np.linalg.norm(prev_to_target) - norm_to_target

            # If the episode is over, check the position of the floor projection
            # of the projectile, to know wether if it's in the trash or not
            if self.episode_over:
                if norm_to_target <= 0.115 and not self.episode_failed:
                    reward += 2.0
                else:
                    self.episode_failed = True

                initial_proj_footprint = [
                    self.initial_projectile_pose[0],
                    self.initial_projectile_pose[1],
                    0.0]

                initial_norm = np.linalg.norm(
                    np.array(initial_proj_footprint) - np.array(bucket_footprint))

                # Test replace norm
                reward += initial_norm - norm_to_target

            # Test velocity vector reward
            # K = 500

            # if (norm_to_target <= 0.115 and projectile_pose[2] <= 0.32):
            #     reward += 1/K
            # elif np.linalg.norm(init_to_current) != 0:
            #     ref_cos = np.cos(np.arctan2(0.3, norm_to_target))
            #     center_cos =\
            #         np.dot(current_to_target, init_to_current) /\
            #         (norm_to_target * np.linalg.norm(init_to_current))

            #     if center_cos > ref_cos:
            #         reward += 1/K
            #     else:
            #         reward += center_cos/K
            # else:
            #     reward += -1/K

            # Add the reward to the episode reward
            self.episode_reward += reward

            # Update the previous projectile pose
            self.prev_projectile_pose = projectile_pose

            if self.episode_over:
                self.episode_number += 1
                self._printEpisodeSummary()

            return obs, reward




            






