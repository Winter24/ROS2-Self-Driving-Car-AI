#git push from raspberry pi
#Control Variables for 3c_threaded_Mod4
import os
import cv2

detect = 1 # Set to 1 for Lane detection

Testing = True# Set to True --> if want to see what the car is seeing
Profiling = False # Set to True --> If you want to profile code
write = False # Set to True --> If you want to Write input / output videos
In_write = False
Out_write = False

debugging = True # Set to True --> If you want to debug code

debugging_Lane = True

debugging_L_ColorSeg = True
debugging_L_Est= True
debugging_L_Cleaning= True
debugging_L_LaneInfoExtraction= True

debugging_Signs = True
debugging_TrafficLights = True
debugging_TL_Config = True
# Adding functionality to toggle Sat_Nav on/off
enable_SatNav = False

# In GPS Nav mode, front-camera road following owns steering whenever the lane
# mask is valid. GPS only supplies route progress and fallback when vision is
# lost, so it cannot fight the camera correction on straight lane-centering.
gps_lane_assist_enabled = True
gps_lane_assist_weight = 1.00
gps_lane_assist_gps_bias_weight = 0.0
gps_lane_assist_max_angular = 0.55
gps_lane_assist_disable_above_gps_turn = 99.0
gps_lane_assist_speed_cap = 0.50
gps_lane_assist_camera_speed = 0.80
gps_lane_assist_gps_fallback_speed = 0.25
gps_nav_max_angular_cmd = 1.00

# Camera road-centering sensitivity. The previous controller used half of the
# image width, making even a clear road-center error produce a tiny correction.
lane_center_max_dist_px = 70
lane_distance_weight = 0.80
lane_curvature_weight = 0.20
road_min_width_ratio = 0.35

# Black-road segmentation thresholds for city GPS lane assist. Use a strict
# dark+low-saturation mask so gray curbs and yellow/green shoulders are not
# treated as drivable asphalt.
road_black_max_hls_l = 62
road_black_max_hls_s = 110
road_black_max_hsv_v = 85
road_black_max_hsv_s = 95

# [NEW]: Control switch to turn steering animation on/off
animate_steering = False

# [NEW]: Containers to store the orignal vs Smoothed steering angle for visualizing the effect
angle_orig = 0
angle = 0
# adding engines on/off control 
engines_on = False
# adding clr_seg_dbg control to create trackbars only once 
clr_seg_dbg_created = False

Detect_lane_N_Draw = True
Training_CNN = False 

vid_path = os.path.abspath("data/vids/Ros2/lane.avi")
loopCount=0


Resized_width = 320#320#240#640#320 # Control Parameter
Resized_height = 240#240#180#480#240

# Create video writers only when recording is enabled. Creating them unconditionally
# can make OpenCV try the image-sequence backend and print warnings when the
# output directory does not exist or writing is disabled.
os.makedirs(os.path.abspath("data/Output"), exist_ok=True)
in_q = (
    cv2.VideoWriter(os.path.abspath("data/Output/in_new.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (Resized_width, Resized_height))
    if In_write else None
)
out = (
    cv2.VideoWriter(os.path.abspath("data/Output/out_new.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), 30, (Resized_width, Resized_height))
    if Out_write else None
)

if debugging:
    waitTime = 1
else:
    waitTime = 1

#============================================ Paramters for Lane Detection =======================================
Ref_imgWidth = 1920
Ref_imgHeight = 1080

#Ref_imgWidth = 640
#Ref_imgHeight = 480

Frame_pixels = Ref_imgWidth * Ref_imgHeight

Resize_Framepixels = Resized_width * Resized_height

Lane_Extraction_minArea_per = 1000 / Frame_pixels
minArea_resized = int(Resize_Framepixels * Lane_Extraction_minArea_per)

BWContourOpen_speed_MaxDist_per = 500 / Ref_imgHeight
MaxDist_resized = int(Resized_height * BWContourOpen_speed_MaxDist_per)

CropHeight = 650 # Required in Camera mounted on top of car 640p
CropHeight_resized = int( (CropHeight / Ref_imgHeight ) * Resized_height )
