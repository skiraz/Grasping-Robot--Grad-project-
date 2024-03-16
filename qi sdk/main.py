import qi 
import os
import numpy as np
import cv2
from naoqi import ALProxy 
import time
import sys
from functions  import *

port = 9559
ip = "192.168.0.102"
session =connect(ip,port)




# standup(session)
# time.sleep(2)
get_depth_image(session)
time.sleep(2)
# move_head(session)


# tablet_service.wakeUp()
print(7656587)
sys.exit()








video_service = session.service("ALVideoDevice")

# set parameters for depth image
resolution = 2  # VGA resolution
color_space = 11  # depth image in mm
fps = 15  # frame rate
video_client_name = "python_client"  # name of the client
depth_camera_index = 1 # index of the depth camera
video_service.subscribeCamera(video_client_name, depth_camera_index, resolution, color_space, fps)

# get depth image
image = None
while image is None:
    image = video_service.getImageRemote(video_client_name)

width, height = image[0], image[1]
depth_array = np.frombuffer(image[6], dtype=np.uint8).reshape([height, width, 2])[:, :, 0]  # convert to numpy array

# release video device
video_service.unsubscribe(video_client_name)

# show depth image
depth_array = depth_array / np.max(depth_array)  # normalize to [0, 1] range
depth_array = (depth_array * 255).astype(np.uint8)  # convert to 8-bit grayscale
cv2.imshow("Depth Image", depth_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
filename = "pepper_image.jpg"
cv2.imwrite(filename, depth_array)










# # set parameters for depth image
# resolution = 2  # VGA resolution
# fps = 15  # frame rate
# depth_service.setResolution(resolution)
# depth_service.setFps(fps)

# # start depth camera
# depth_service.startDepthCam()

# # get depth image
# depth_image = depth_service.getDepthMap()
# depth_array = np.asarray(depth_image)  # convert to numpy array

# # stop depth camera
# depth_service.stopDepthCam()

# # show depth image
# depth_array = depth_array / np.max(depth_array)  # normalize to [0, 1] range
# depth_array = (depth_array * 255).astype(np.uint8)  # convert to 8-bit grayscale
# cv2.imshow("Depth Image", depth_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









# sys.exit()



# mot = session.service("ALMotion")
# cam = session.service("ALVideoDevice")
# print(cam.hasDepthCamera())
# mot.setStiffnesses("Head",1)
# names = "HeadPitch"
# angle = 0.5
# f = 0.1

# mot.setAngles(names,angle,f)
# # print(34242342)

# # time.sleep(2)
# # mot.setStiffnesses("Head",0)






# # # Get a reference to the camera service
# camera_service = session.service("ALVideoDevice")

# # Set the camera parameters
# resolution = 11 # 320x240
# color_space = 1 # RGB
# fps = 10

# # Start the camera
# camera_id = camera_service.subscribeCamera("python_client", 0, resolution, color_space, fps)

# # Capture an image
# image = camera_service.getImageRemote(camera_id)
# print(image)

# # Convert the image data to a Numpy array
# image_width = image[0]
# image_height = image[1]
# # print(image[6])
# image_array = np.frombuffer(image[6], dtype=np.uint8).reshape((image_height, image_width, 3))

# # Save the image to a file on your laptop
# filename = "pepper_image.jpg"
# cv2.imwrite(filename, image_array)

# # Stop the camera
# camera_service.unsubscribe(camera_id)

