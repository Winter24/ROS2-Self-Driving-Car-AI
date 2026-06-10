


debug = False

debug_localization = False
debug_mapping = False
debug_pathplanning = False
debug_motionplanning = False

debug_live = False
debug_live_amount = 0
debug_map_live_amount = 0
debug_path_live_amount = 0

# [NEW]: Container to store destination_pt selected by User
destination = []

# Manual override for GPS start point in road-network frame (x, y).
# Set to None to use visual localization.
manual_start = (759, 359)


# Show/save step-by-step GPS planner debug images.
visualize_pipeline = True
visualize_pipeline_dir = "/tmp/gps_nav_debug"

# Manual GPS odom controller tuning.
# gps_meters_per_pixel maps waypoint-map pixels to Gazebo /odom meters.
# If the car stops too early/late, tune this value first.
gps_meters_per_pixel = 0.14
# Manual waypoint nodes are on the road graph, not necessarily in the lane
# center. Shift the followed GPS path to the left of travel by this many pixels
# so the car does not ride the curb/road edge. Tune this first if it drives too
# close to the lane border.
gps_lane_offset_px = 0.0
gps_waypoint_spacing_px = 35.0
# Extra yaw offset used to generate the /odom control path. Keep this at 0 when
# /odom yaw already points along the car nose; node 13 -> 12 should then
# generate a straight path instead of a left/right lateral target.
gps_odom_forward_yaw_offset_rad = 0.0

# Separate offset used only to draw/project the odom car marker back onto the
# GPS/SatView map. This can differ from the control offset because the camera
# map orientation and the Gazebo odom control frame are not necessarily the same.
gps_odom_display_yaw_offset_rad = -1.5707963267948966
gps_odom_goal_tolerance_m = 1.2
gps_odom_min_linear = 0.35
gps_odom_max_linear = 1.4
gps_odom_angular_kp = 1.8
gps_odom_max_angular = 1.8
