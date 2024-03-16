
import numpy as np
import time
import pybullet as p
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet_data
import cv2
from qibullet import Camera
from qibullet import PepperVirtual
from qibullet.camera import CameraRgb
from qibullet.camera import CameraDepth
import threading
# version = p.getApiVersion()





if __name__ == "__main__":
    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True, auto_step=True)
    robot = simulation_manager.spawnPepper(client, spawn_ground_plane=True)
    
    gravity_x = 0
    gravity_y = 0
    gravity_z = -4.81  # Adjust the value as per your requirements
    p.setGravity(gravity_x, gravity_y, gravity_z)

    robot.goToPosture("sit",percentage_speed=1)

    time.sleep(3)
    table_visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.4, 0.4, 0.02],
        rgbaColor=[0.9, 0.9, 0.9, 1.0])
    
    table_collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.4, 0.4, 0.02])
    
    table_position = [0.6, -.1, .8  ]  # Modify this position to place the table where you want
    
    table_body_id = p.createMultiBody(
        baseMass=0.0,
        baseInertialFramePosition=[0.0, 0.0, 0.0],
        baseCollisionShapeIndex=table_collision_shape_id,
        baseVisualShapeIndex=table_visual_shape_id,
        basePosition=table_position)
    


    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    cube = p.loadURDF(
    r"D:\projects\graduation project\qi bullet simulation\soda_can_tex.urdf",
    basePosition=[0.35, 0.1, .96],
    globalScaling=0.7,
    physicsClientId=client
)

# Load and apply the Coca-Cola can texture
    texture_id = p.loadTexture(r"D:\projects\graduation project\qi bullet simulation\pepsi.png")
    p.changeVisualShape(cube, -1, textureUniqueId=texture_id)

    robot_body_id = robot.robot_model
    print(robot_body_id,cube)
   
    # constraint_id = p.createConstraint(
    #     parentBodyUniqueId=robot_body_id,
    #     parentLinkIndex=robot.link_dict["l_gripper"].getIndex(),
    #     childBodyUniqueId=cube,
    #     childLinkIndex=0,
    #     jointType=p.JOINT_FIXED,
    #     jointAxis=(0, 0, 0),
    #     parentFramePosition=(0, 0, 0),
    #     childFramePosition=(0, 0, 0)
    # )




    # Enable texture rendering
    # p.configureDebugVisualizer(p.COV_ENABLE_TEXTURE_RENDERING, 1)
    # print("waiting for rotation")
    # time.sleep(2)
    # print("rotating")
    # robot.moveTo(0, 0, -2, speed=1)
    # print("rotating finished !")
    # time.sleep(1)
    
    joint_parameters = list()

    for name, joint in robot.joint_dict.items():
        if "Finger" not in name and "Thumb" not in name:
            joint_parameters.append((
                p.addUserDebugParameter(
                    name,
                    joint.getLowerLimit(),
                    joint.getUpperLimit(),
                    robot.getAnglesPosition(name)),     
                name))
            
    # for i in range(33,48):
    #    p.setJointMotorControl2(
    #     bodyUniqueId=robot_body_id,
    #     jointIndex=i,
    #     controlMode=p.TORQUE_CONTROL,  # or p.VELOCITY_CONTROL
    #     force=100) 
    
    # p.changeDynamics(cube, -1, localInertiaDiagonal=0.6)
    
    
    '''this the normal but im trying to make better versions of it,'''

    # def camera_view(robot):
    #     taken = False
    #     while True:
    #         handle_depth = robot.subscribeCamera(
    #             PepperVirtual.ID_CAMERA_DEPTH,
    #             resolution=Camera.K_QVGA,
    #             fps=20)

    #         depth_image = robot.getCameraFrame(handle_depth)
    #         # depth_image = np.reshape(img_depth[3], (img_depth[1], img_depth[0]))

    #         # Normalize the depth values
    #         # normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    #         # Apply a color map to the depth image
    #         # depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)


    #         depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

    #         # Invert the depth values (so closer objects are lighter)
    #         depth_image_inverted = 255 - depth_image_normalized

    #         # Apply a color map to the depth image
    #         depth_colormap = cv2.applyColorMap(depth_image_inverted.astype(np.uint8), cv2.COLORMAP_JET)


    #         cv2.imshow("Depth Camera", depth_colormap)
    #         cv2.waitKey(1)



    def camera_view(robot):
        taken = False
        while True:
            handle_depth = robot.subscribeCamera(
                PepperVirtual.ID_CAMERA_TOP,
                resolution=Camera.K_QVGA,
                fps=20)

            depth_image = robot.getCameraFrame(handle_depth)
            # depth_image = np.reshape(img_depth[3], (img_depth[1], img_depth[0]))

            # Normalize the depth values
            # normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # Apply a color map to the depth image
            # depth_colormap = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
            # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # # Invert the depth values (so closer objects are lighter)
            # depth_image_inverted = 255 - depth_image_normalized

            # # Apply a square root mapping to the depth values
            # depth_image_sqrt = np.sqrt(depth_image_inverted)

            # # Normalize the mapped depth image to 0-255 range
            # depth_image_normalized = cv2.normalize(depth_image_sqrt, None, 0, 255, cv2.NORM_MINMAX)

            # # Convert the depth image to 8-bit unsigned integer format
            # depth_image_uint8 = depth_image_normalized.astype(np.uint8)

            # # Apply a color map to the depth image
            # depth_colormap = cv2.applyColorMap(depth_image_uint8, cv2.COLORMAP_JET)


            # # depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)

            # # # Invert the depth values (so closer objects are lighter)
            # # depth_image_inverted = 255 - depth_image_normalized

            # # # Apply a color map to the depth image
            # # depth_colormap = cv2.applyColorMap(depth_image_inverted.astype(np.uint8), cv2.COLORMAP_JET)


            cv2.imshow("RGB", depth_image)
            
            cv2.imwrite("rgb.jpg",depth_image)
            
            cv2.waitKey(1)



    
    

    

 
    # Check if the link index was found
    def collision():
        contact_points = p.getContactPoints(robot_body_id, cube)
        
        if len(contact_points) > 0:
            print(contact_points)
            print("Collision detected!")
    

    def _getLinkPosition( link_name):
                """
                Returns the position of the link in the world frame
                """
                link_state = p.getLinkState(
                    robot_body_id,
                    robot.link_dict[link_name].getIndex())
                
                return link_state[0], link_state[1]
    # camera thread 
    camera_thread = threading.Thread(target=camera_view,args=(robot,))
    camera_thread.start()
    # for i in range(p.getNumJoints(cube, physicsClientId=client)):
    #             print(p.getLinkState(cube, i, physicsClientId=client))

    # print(p.getNumJoints(cube, physicsClientId=client))

    try:
        while True:
            
           
            for joint_parameter in joint_parameters:
                robot.setAngles(
                    joint_parameter[1],
                    p.readUserDebugParameter(joint_parameter[0]), 1)
            #     table_position = p.getBasePositionAndOrientation(table_body_id)[0]
            #     soda_can_position = [table_position[0], table_position]
            


            # # Loop over all links in the robot
            # for link_id in list(range(36,50,1)):
            #     # Check if there are any contact points between the link and the cube
            #     contact_points = p.getContactPoints(robot_body_id, cube, linkIndexA=link_id)
            #     if len(contact_points) > 0:
            #         print("Collision detected between the ball and link", link_id)

            # for link_id in range(p.getNumJoints(robot_body_id)):
            #     # Check if there are any contact points between the link and the ball
            #     contact_points = p.getContactPoints(robot_body_id, table_body_id, linkIndexA=link_id)
            #     if len(contact_points) > 0:
            #         print("Collision detected between the table and link", link_id)
            # # camera_view(robot)



            

            # for link_id in hand_link_ids:
            #     # Check if there are any contact points between the hand link and the ball link
            #     contact_points = p.getContactPoints(robot.robot_model, ball_link_id, linkIndexA=link_id)
            #     print(contact_points)
            #     if len(contact_points) > 0:
            #         print("yes")
            #         break


            
            # print(_getLinkPosition("l_gripper"))

            # print( p.getBasePositionAndOrientation(cube)[0])
            collision()
                

            link_state = p.getLinkState(
            robot.robot_model,
            robot.link_dict["l_gripper"].getIndex())
        
            SodaPos = p.getBasePositionAndOrientation(cube)[0]
            linkPos = link_state[0]
            distance = np.sqrt(np.sqrt( (SodaPos[0]-linkPos[0])**2+(SodaPos[1]-linkPos[1])**2+(SodaPos[2]+0.15-linkPos[2])**2 ))
            # print(distance)



            


            # Step the simulation
            simulation_manager.stepSimulation(client)

    except KeyboardInterrupt:
        pass
    finally:
        simulation_manager.stopSimulation(client)