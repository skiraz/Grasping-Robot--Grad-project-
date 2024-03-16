import qi 
from naoqi import ALProxy 
import vision_definitions
import numpy as np
import cv2


def standup(session):
    motion_service = session.service("ALMotion")
    posture_service = session.service("ALRobotPosture")
    posture_service.goToPosture("StandInit",0.5)
    # motion_service.rest()
    # motion_service.wakeUp()
    return 


def move_head(session):
    mot = session.service("ALMotion")
    mot.setStiffnesses("Head",1)
    names = "HeadPitch"
    angle = 0.5
    f = 0.1

    mot.setAngles(names,angle,f)



def get_depth_image(session):
    camera_service = session.service("ALVideoDevice")

    # Set the camera parameters
    resolution = 1 # 320x240
    color_space =0 # RGB
    fps = 10

    # Start the camera
    camera_id = camera_service.subscribeCamera("python_client", 2, resolution, color_space, fps)
    # Capture an image
    image = camera_service.getImageRemote(camera_id)
    # print(image)

    # Convert the image data to a Numpy array
    image_width = image[0]
    image_height = image[1]
    # print(image[6])
    image_array = np.frombuffer(image[6], dtype=np.uint8).reshape((image_height, image_width, 1))

    # Save the image to a file on your laptop
    filename = "pepper_image.jpg"
    cv2.imwrite(filename, image_array)

    # Stop the camera
    camera_service.unsubscribe(camera_id)


def connect(ip,port):
    session = qi.Session()
    
    connection_url = "tcp://" + ip + ":" + str(port)
    session.connect(connection_url)
    return session
