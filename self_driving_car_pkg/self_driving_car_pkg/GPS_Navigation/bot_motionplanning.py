'''
> Purpose :
Module to perform motionplanning for helping the vehicle navigate to the desired destination

> Usage :
You can perform motionplanning by
1) Importing the class (bot_motionplanner)
2) Creating its object
3) Accessing the object's function of (nav_path). 
E.g ( self.bot_motionplanner.nav_path(bot_loc, path, self.vel_msg, self.velocity_publisher) )


> Inputs:
1) Robot Current location
2) Found path to destination
3) Velocity object for manipulating linear and angular component of robot
4) Velocity publisher to publish the updated velocity object

> Outputs:
1) speed              => Speed with which the car travels at any given moment
2) angle              => Amount of turning the car needs to do at any moment

Author :
Haider Abbasi

Date :
6/04/22
'''
import cv2
import numpy as np
from math import pow , atan2,sqrt , degrees,asin, sin, cos, pi

from numpy import interp
import pygame
from ament_index_python.packages import get_package_share_directory
import os
pygame.mixer.init()

from . import config

class bot_motionplanner():


    def __init__(self):

        # counter to move car forward for a few iterations
        self.count = 0
        # State Variable => Initial Point Extracted?
        self.pt_i_taken = False
        # [Container] => Store Initial car location
        self.init_loc = 0

        # State Variable => Angle relation computed?
        self.angle_relation_computed = False

        # [Container] => Bot Angle [Image]
        self.bot_angle = 0
        # [Container] => Bot Angle [Simulation]
        self.bot_angle_s = 0
        # [Container] => Angle Relation Bw(Image & Simulation)
        self.bot_angle_rel = 0
        # State Variable ==> (Maze Exit) Not Reached ?
        self.goal_not_reached_flag = True
        # [Containers] ==> Mini-Goal (X,Y)
        self.goal_pose_x = 0
        self.goal_pose_y = 0
        # [Iterater] ==> Current Mini-Goal iteration
        self.path_iter = 0

        # [NEW]: Delete Not required variables
        # [NEW]: Modify curr_speed -> req_speed and angle
        self.req_speed = 0
        self.req_angle = 0

        # [New]: Booelean to note when car is taking a sharp turn
        # Handy when encountering turn at dcsn poitns
        self.car_turning = False

        # [New]: Containers to store vel and angle needed to published to velicity 
        # publisher and instead are saved here .
        # And decsn will be based on taking both info into account)
        self.vel_linear_x = 1.0
        self.vel_angular_z = 0.0

        # [New]: Contaienrs to save speed and angle of car provided by sensors on motors
        self.actual_speed = 0
        self.actual_angle = 0

        # Stable GPS navigation state: use /odom x/y/yaw instead of visual
        # background-subtraction localization. Waypoints are converted from
        # GPS-map pixels into an odom-relative world path when the route is
        # selected.
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw_rad = 0.0
        self.odom_ready = False
        self.odom_path = []
        self.odom_path_sat = []
        self.odom_path_iter = 0
        self._sat_to_world_cfg = None
        self.odom_goal_tolerance_m = float(getattr(config, "gps_odom_goal_tolerance_m", 1.2))
        self.odom_max_linear = float(getattr(config, "gps_odom_max_linear", 1.4))
        self.odom_min_linear = float(getattr(config, "gps_odom_min_linear", 0.35))
        self.odom_angular_kp = float(getattr(config, "gps_odom_angular_kp", 1.8))
        self.odom_max_angular = float(getattr(config, "gps_odom_max_angular", 1.8))
        self.gps_meters_per_pixel = float(getattr(config, "gps_meters_per_pixel", 0.10))
        self.odom_forward_yaw_offset = float(getattr(config, "gps_odom_forward_yaw_offset_rad", 0.0))
        self.odom_display_yaw_offset = float(getattr(config, "gps_odom_display_yaw_offset_rad", 0.0))
        self.gps_lane_offset_px = float(getattr(config, "gps_lane_offset_px", 0.0))
        self.gps_waypoint_spacing_px = float(getattr(config, "gps_waypoint_spacing_px", 35.0))


    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians
  
    def get_pose(self,data):

        # We get the bot_turn_angle in simulation Using same method as Gotogoal.py
        quaternions = data.pose.pose.orientation
        (roll,pitch,yaw)=self.euler_from_quaternion(quaternions.x, quaternions.y, quaternions.z, quaternions.w)
        yaw_deg = degrees(yaw)
        self.pose_x = float(data.pose.pose.position.x)
        self.pose_y = float(data.pose.pose.position.y)
        self.yaw_rad = float(yaw)
        self.odom_ready = True

        # [Maintaining the Consistency in Angle Range]
        if (yaw_deg>0):
            self.bot_angle_s = yaw_deg
        else:
            # -160 + 360 = 200, -180 + 360 = 180 . -90 + 360 = 270
            self.bot_angle_s = yaw_deg + 360
        
        #              Bot Rotation 
        #      (OLD)        =>      (NEW) 
        #   [-180,180]             [0,360]
        # [NEW]: 2) Retrieving Bot Current Speed from its odometry measurements
        # We get the bot_turn_angle in simulation Using same method as Gotogoal.py
        self.actual_speed = -(data.twist.twist.linear.x)

        if self.actual_speed<0.005:
            self.actual_speed = 0.00

        self.actual_angle = data.twist.twist.angular.z

    @staticmethod
    def _wrap_to_pi(angle):
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle

    @staticmethod
    def _sat_vec_to_map_vec(a, b):
        """Convert image-pixel vector a->b into Cartesian map vector.

        SatView has x right, y down. For control we use x right, y up, so dy is
        negated.
        """
        return (float(b[0]) - float(a[0]), float(a[1]) - float(b[1]))

    def _prepare_sat_path_for_control(self, sat_path):
        """Densify and lane-shift the selected SatView node path.

        Manual waypoint nodes mark the road graph. They can sit near the road
        edge, so following them directly makes the car drift toward the curb.
        Positive gps_lane_offset_px shifts the path to the left of travel in
        image coordinates; negative shifts right.
        """
        if len(sat_path) < 2:
            return sat_path

        spacing = max(1.0, self.gps_waypoint_spacing_px)
        offset = self.gps_lane_offset_px
        dense = []
        for i in range(len(sat_path) - 1):
            ax, ay = sat_path[i]
            bx, by = sat_path[i + 1]
            dx = bx - ax
            dy = by - ay
            dist = sqrt(dx * dx + dy * dy)
            if dist < 1e-6:
                continue
            # Image-space left normal for travel vector (dx, dy).
            nx = dy / dist
            ny = -dx / dist
            steps = max(1, int(dist / spacing))
            for j in range(steps):
                if i > 0 and j == 0:
                    continue
                t = float(j) / float(steps)
                dense.append((ax + t * dx + offset * nx, ay + t * dy + offset * ny))
        # Keep the final shifted endpoint.
        ax, ay = sat_path[-2]
        bx, by = sat_path[-1]
        dx = bx - ax
        dy = by - ay
        dist = max(1e-6, sqrt(dx * dx + dy * dy))
        dense.append((bx + offset * dy / dist, by - offset * dx / dist))
        return dense if len(dense) >= 2 else sat_path

    def configure_odom_path_from_sat(self, sat_path):
        """Create an odom/world waypoint path from selected SatView nodes.

        Calibration is intentionally simple and stable:
        - the selected START node is anchored to current /odom x/y;
        - the first route segment is aligned with the car's current yaw;
        - pixel distances are scaled by config.gps_meters_per_pixel.

        This avoids visual object detection completely. If route scale is too
        long/short, tune config.gps_meters_per_pixel.
        """
        if not self.odom_ready:
            print("[GPS Odom Debug] waiting for /odom before configuring path")
            return False
        if sat_path is None or len(sat_path) < 2:
            print("[GPS Odom Debug] invalid sat path")
            return False

        sat_path = [tuple(map(float, p)) for p in sat_path]
        sat_path_raw = sat_path
        sat_path = self._prepare_sat_path_for_control(sat_path_raw)
        start = sat_path[0]
        # Find first non-zero segment for orientation.
        first_vec = None
        for p in sat_path[1:]:
            v = self._sat_vec_to_map_vec(start, p)
            if sqrt(v[0] * v[0] + v[1] * v[1]) > 1.0:
                first_vec = v
                break
        if first_vec is None:
            print("[GPS Odom Debug] sat path has no non-zero segment")
            return False

        first_angle_map = atan2(first_vec[1], first_vec[0])
        # /odom yaw can be tied to the model link frame, not the car's visible
        # forward direction on the satellite map. Calibrate path rotation with
        # the configured forward-heading offset so a straight route remains
        # straight in the GPS overlay.
        car_forward_yaw = self._wrap_to_pi(self.yaw_rad + self.odom_forward_yaw_offset)
        rot = car_forward_yaw - first_angle_map
        scale = self.gps_meters_per_pixel
        origin_world = (self.pose_x, self.pose_y)
        c = cos(rot)
        ss = sin(rot)

        world_path = []
        for p in sat_path:
            mx, my = self._sat_vec_to_map_vec(start, p)
            wx = origin_world[0] + scale * (c * mx - ss * my)
            wy = origin_world[1] + scale * (ss * mx + c * my)
            world_path.append((wx, wy))

        self.odom_path = world_path
        self.odom_path_sat = sat_path
        self.odom_path_iter = 1 if len(world_path) > 1 else 0
        self.goal_not_reached_flag = True
        self.vel_linear_x = 0.0
        self.vel_angular_z = 0.0
        display_rot = self._wrap_to_pi(rot + self.odom_display_yaw_offset - self.odom_forward_yaw_offset)
        self._sat_to_world_cfg = {
            "origin_sat": start,
            "origin_world": origin_world,
            "rot": rot,
            "display_rot": display_rot,
            "scale": scale,
        }
        print("[GPS Odom Debug] configured odom path: scale={:.3f}m/px lane_offset={:.1f}px yaw={:.1f}deg control_yaw={:.1f}deg display_yaw={:.1f}deg first_map_angle={:.1f}deg nodes={}".format(
            scale, self.gps_lane_offset_px, degrees(self.yaw_rad), degrees(car_forward_yaw),
            degrees(self.yaw_rad + self.odom_display_yaw_offset), degrees(first_angle_map), len(world_path)))
        print("[GPS Odom Debug] world path =", [tuple(round(v, 2) for v in p) for p in world_path])
        return True

    def world_to_sat(self, world_xy):
        """Project current odom/world point back to SatView for visualization.

        Forward transform (configure_odom_path_from_sat):
            map_vec  = _sat_vec_to_map_vec(start, p)
                     = (p_sat_x - start_x,  start_y - p_sat_y)   # y-flip: sat-y-down -> map-y-up
            world_pt = origin_world + scale * R(rot) * map_vec

        Inverse:
            map_vec  = R(-rot) * (world_pt - origin_world) / scale
            sat_x    = start_x + map_vec_x
            sat_y    = start_y - map_vec_y                        # undo y-flip
        """
        cfg = self._sat_to_world_cfg
        if not cfg:
            return None
        ox, oy = cfg["origin_world"]
        sx, sy = cfg["origin_sat"]
        scale = cfg["scale"]
        rot = cfg.get("display_rot", cfg["rot"])
        if abs(scale) < 1e-9:
            return None
        # world delta
        wdx = (float(world_xy[0]) - ox) / scale
        wdy = (float(world_xy[1]) - oy) / scale
        # inverse rotation R(-rot)
        cr = cos(-rot)
        sr = sin(-rot)
        mx = cr * wdx - sr * wdy
        my = sr * wdx + cr * wdy
        # undo y-flip: sat_y = origin_sat_y - map_y
        return (int(round(sx + mx)), int(round(sy - my)))

    def nav_path_odom(self, world_path=None):
        """Follow waypoint path using /odom x/y/yaw only.

        This replaces the old image-space controller for manual GPS mode.
        """
        if world_path is not None:
            self.odom_path = world_path
        path = self.odom_path
        if not self.odom_ready:
            self.vel_linear_x = 0.0
            self.vel_angular_z = 0.0
            print("[GPS Odom Debug] no /odom yet; stopping")
            return
        if path is None or len(path) < 2:
            self.vel_linear_x = 0.0
            self.vel_angular_z = 0.0
            return
        if not self.goal_not_reached_flag:
            self.vel_linear_x = 0.0
            self.vel_angular_z = 0.0
            return

        self.odom_path_iter = max(1, min(self.odom_path_iter, len(path) - 1))
        tx, ty = path[self.odom_path_iter]
        dx = tx - self.pose_x
        dy = ty - self.pose_y
        dist = sqrt(dx * dx + dy * dy)

        # Advance waypoint when close enough OR when the car has passed the
        # target along the current path segment. Camera-dominant control can run
        # slightly beside the GPS path, so a pure distance-to-waypoint check may
        # never enter the small radius and the car overshoots the destination.
        prev_x, prev_y = path[self.odom_path_iter - 1]
        seg_x = tx - prev_x
        seg_y = ty - prev_y
        seg_len2 = (seg_x * seg_x) + (seg_y * seg_y)
        if seg_len2 > 1e-9:
            along_t = (((self.pose_x - prev_x) * seg_x) + ((self.pose_y - prev_y) * seg_y)) / seg_len2
        else:
            along_t = 0.0

        if dist <= self.odom_goal_tolerance_m or along_t >= 1.0:
            if self.odom_path_iter >= len(path) - 1:
                self.goal_not_reached_flag = False
                self.vel_linear_x = 0.0
                self.vel_angular_z = 0.0
                print("[GPS Odom Debug] destination reached/passed at odom ({:.2f},{:.2f}) dist={:.2f} along={:.2f}".format(
                    self.pose_x, self.pose_y, dist, along_t))
                try:
                    pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'resource', 'Goal_reached.wav'))
                    pygame.mixer.music.play()
                except Exception:
                    pass
                return
            self.odom_path_iter += 1
            tx, ty = path[self.odom_path_iter]
            dx = tx - self.pose_x
            dy = ty - self.pose_y
            dist = sqrt(dx * dx + dy * dy)
            print("[GPS Odom Debug] next waypoint {}/{} world=({:.2f},{:.2f})".format(
                self.odom_path_iter, len(path)-1, tx, ty))

        target_yaw = atan2(dy, dx)
        yaw_error = self._wrap_to_pi(target_yaw - self.yaw_rad)
        ang = max(-self.odom_max_angular, min(self.odom_max_angular, self.odom_angular_kp * yaw_error))

        # Slow down when heading error is large; do not drive fast while turning around.
        heading_scale = max(0.15, 1.0 - min(abs(yaw_error), pi) / pi)
        lin = min(self.odom_max_linear, max(self.odom_min_linear, 0.45 * dist)) * heading_scale
        if abs(yaw_error) > 1.2:
            lin = 0.15

        self.vel_linear_x = float(lin)
        self.vel_angular_z = float(ang)
        self.car_turning = abs(yaw_error) > 0.25
        self.goal_pose_x = tx
        self.goal_pose_y = ty

        if getattr(config, "debugging", False) or getattr(config, "debug_motionplanning", False):
            print("[GPS Odom Debug] pose=({:.2f},{:.2f},{:.1f}deg) target[{}/{}]=({:.2f},{:.2f}) dist={:.2f} yaw_err={:.1f}deg cmd=({:.2f},{:.2f})".format(
                self.pose_x, self.pose_y, degrees(self.yaw_rad), self.odom_path_iter, len(path)-1,
                tx, ty, dist, degrees(yaw_error), self.vel_linear_x, self.vel_angular_z))

    @staticmethod
    def bck_to_orig(pt,transform_arr,rot_mat):

        st_col = transform_arr[0] # cols X
        st_row = transform_arr[1] # rows Y
        tot_cols = transform_arr[2] # total_cols / width W
        tot_rows = transform_arr[3] # total_rows / height H
        
        # point --> (col(x),row(y)) XY-Convention For Rotation And Translated To MazeCrop (Origin)
        #pt_array = np.array( [pt[0]+st_col, pt[1]+st_row] )
        pt_array = np.array( [pt[0], pt[1]] )
        
        # Rot Matrix (For Normal XY Convention Around Z axis = [cos0 -sin0]) But for Image convention [ cos0 sin0]
        #                                                      [sin0  cos0]                           [-sin0 cos0]
        rot_center = (rot_mat @ pt_array.T).T# [x,y]
        
        # Translating Origin If neccasary (To get whole image)
        rot_cols = tot_cols#tot_rows
        rot_rows = tot_rows#tot_cols
        rot_center[0] = rot_center[0] + (rot_cols * (rot_center[0]<0) ) + st_col  
        rot_center[1] = rot_center[1] + (rot_rows * (rot_center[1]<0) ) + st_row 
        return rot_center

    def display_control_mechanism_in_action(self,bot_loc,path,img_shortest_path,bot_localizer,frame_disp):
        Doing_pt = 0
        Done_pt = 0

        # No valid path: keep UI alive and do not index into path.
        if type(path) == int or path is None or len(path) < 2:
            if img_shortest_path is not None and hasattr(img_shortest_path, 'shape'):
                img_shortest_path = cv2.circle(img_shortest_path, bot_loc, 3, (0,0,255))
            st = "GPS path unavailable"
            frame_disp = cv2.putText(frame_disp, st, (bot_localizer.orig_X+50,bot_localizer.orig_Y-30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
            if config.debug and config.debug_motionplanning and img_shortest_path is not None and hasattr(img_shortest_path, 'shape'):
                cv2.imshow("maze (Shortest Path + Car Loc)",img_shortest_path)
            return

        path_i = min(self.path_iter, len(path)-1)
        
        # Circle to represent car current location
        img_shortest_path = cv2.circle(img_shortest_path, bot_loc, 3, (0,0,255))

        if ( path_i!=(len(path)-1) ):
            curr_goal = path[path_i]
            # Mini Goal Completed
            if path_i!=0:
                img_shortest_path = cv2.circle(img_shortest_path, path[path_i-1], 3, (0,255,0),2)
                Done_pt = path[path_i-1]
            # Mini Goal Completing   
            img_shortest_path = cv2.circle(img_shortest_path, curr_goal, 3, (0,140,255),2)
            Doing_pt = curr_goal
        else:
            # Only Display Final Goal completed
            img_shortest_path = cv2.circle(img_shortest_path, path[path_i], 10, (0,255,0))
            Done_pt = path[path_i]

        if Doing_pt!=0:
            Doing_pt = self.bck_to_orig(Doing_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            frame_disp = cv2.circle(frame_disp, (int(Doing_pt[0]),int(Doing_pt[1])), 3, (0,140,255),2)   
            #loc_car_ = self.bck_to_orig(loc_car, bot_localizer_obj.transform_arr, bot_localizer_obj.rot_mat_rev)
            #frame_disp = cv2.circle(frame_disp, (int(loc_car_[0]),int(loc_car_[1])), 3, (0,0,255))
         
            
        if Done_pt!=0:
            Done_pt = self.bck_to_orig(Done_pt, bot_localizer.transform_arr, bot_localizer.rot_mat_rev)
            if ( path_i!=(len(path)-1) ):
                pass
                #frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 3, (0,255,0),2)   
            else:
                frame_disp = cv2.circle(frame_disp, (int(Done_pt[0]),int(Done_pt[1])) , 10, (0,255,0))  

        st = "len(path) = ( {} ) , path_iter = ( {} )".format(len(path),self.path_iter)        
        
        frame_disp = cv2.putText(frame_disp, st, (bot_localizer.orig_X+50,bot_localizer.orig_Y-30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,255))
        if config.debug and config.debug_motionplanning:
            cv2.imshow("maze (Shortest Path + Car Loc)",img_shortest_path)
        else:
            try:
                cv2.destroyWindow("maze (Shortest Path + Car Loc)")
            except:
                pass

    @staticmethod
    def angle_n_dist(pt_a,pt_b):
        # Trignometric rules Work Considering.... 
        #
        #       [ Simulation/Normal Convention ]      [ Image ]
        #
        #                    Y                    
        #                     |                     
        #                     |___                     ____ 
        #                          X                  |     X
        #                                             |
        #                                           Y
        #
        # Solution: To apply same rules , we subtract the (first) point Y axis with (Second) point Y axis
        error_x = pt_b[0] - pt_a[0]
        error_y = pt_a[1] - pt_b[1]

        # Calculating distance between two points
        distance = sqrt(pow( (error_x),2 ) + pow( (error_y),2 ) )

        # Calculating angle between two points [Output : [-Pi,Pi]]
        angle = atan2(error_y,error_x)
        # Converting angle from radians to degrees
        angle_deg = degrees(angle)

        if (angle_deg>0):
            return (angle_deg),distance
        else:
            # -160 +360 = 200, -180 +360 = 180,  -90 + 360 = 270
            return (angle_deg + 360),distance
        
        #             Angle bw Points 
        #      (OLD)        =>      (NEW) 
        #   [-180,180]             [0,360]


    def go_to_goal(self,bot_loc,path):

        # Finding the distance and angle between (current) bot location and the (current) mini-goal
        angle_to_goal,distance_to_goal = self.angle_n_dist(bot_loc, (self.goal_pose_x,self.goal_pose_y))

        # Computing the angle the bot needs to turn to align with the mini goal
        angle_to_turn = angle_to_goal - self.bot_angle

        # [NEW]: Always turning in that direction where it takes less time to realign to goal
        if angle_to_turn>180:
            angle_to_turn = -360 + angle_to_turn
        elif angle_to_turn<-180:
            angle_to_turn =  360 + angle_to_turn

        # [NEW]: Higher Upper Speed limit + Setting speed of bot proportional to its distance to the goal
        speed = interp(distance_to_goal,[0,100],[0.4,2.5])
        self.req_speed = speed
        # Setting steering angle of bot proportional to the amount of turn it is required to take
        angle = interp(angle_to_turn,[-360,360],[-4,4])
        self.req_angle = angle
        
        if (config.debug and config.debug_motionplanning):
            print("angle to goal = {} Angle_to_turn = {} angle[Sim] {}".format(angle_to_goal,angle_to_turn,abs(angle)))
            print("distance_to_goal = ",distance_to_goal)


        # [NEW]: Speed and angle will only be passed in two cases
        #        1) Angle > 15 deg then car is turning sharply
        #                   Go with a set speed and turn as required
        #        2) Destination reached: then speed and gnle to zero
        if self.goal_not_reached_flag:
            if (abs(angle_to_turn)>=15):
                self.car_turning = True

                self.vel_linear_x = 1.0
                self.vel_angular_z = angle
            else:
                self.vel_linear_x = speed
                self.car_turning = False
        else:
            # Stop Car
            self.vel_linear_x = 0.0
            self.vel_angular_z = 0.0

        

        # [NEW]: Updated Reasonable distance + If car is within reasonable distance of mini-goal
        if ((distance_to_goal<=40) ):
                

            self.velocity_linear_x = 0.0
            self.velocity_angular_z = 0.0

            # Reached the final goal
            if self.path_iter==(len(path)-1):
                # [NEW]: Check if not already reahed and within reasomable distance to goal
                if (self.goal_not_reached_flag and (distance_to_goal<=10)):
                    # Set goal_not_reached_flag to False
                    self.goal_not_reached_flag = False
                    # [NEW]: Play the party song, Mention that reached goal
                    pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), 'resource', 'Goal_reached.wav'))
                    pygame.mixer.music.play()
            # Still doing mini-goals?
            else:
                # Iterate over the next mini-goal
                self.path_iter += 1
                self.goal_pose_x = path[self.path_iter][0]
                self.goal_pose_y = path[self.path_iter][1]
                #print("Current Goal (x,y) = ( {} , {} )".format(path[self.path_iter][0],path[self.path_iter][1]))
                

    # [NEW]: Takes on 1 New input bot_loc_wrt_cropping
    #        Removes 2 input as not required anymore
    #               Veloctiy + velicity publisher
    def nav_path(self,bot_loc,bot_loc_wrt_rdnetwrk,path):
        """Performs motionplanning to aid vehicle in navigating to the desired destination

        Args:
            bot_loc              (tuple): Robot Current location
            bot_loc_wrt_rdnetwrk (tuple): Robot Current location adjusted to the road network
            path           (List[tuple]): Found path to destination
        Updates:
            speed              => Speed with which the car travels at any given moment
             angle              => Amount of turning the car needs to do at any moment
        """        
        # If valid path found, choose the active mini-goal.
        # Manual waypoint routes include the selected START node as path[0].
        # When the car is already close to that start node, immediately target
        # path[1]; otherwise the car wastes time/turns back trying to hit the
        # exact clicked start pixel before following the route.
        if type(path) != int and path is not None and len(path) >= 2:
            if self.path_iter == 0:
                _, dist_to_start = self.angle_n_dist(bot_loc, path[0])
                if dist_to_start <= 70:
                    self.path_iter = 1
                    print("[GPS Motion Debug] skip path[0] START; live bot is {:.1f}px away, next target path[1]={}".format(
                        dist_to_start, path[1]))
                self.goal_pose_x = path[self.path_iter][0]
                self.goal_pose_y = path[self.path_iter][1]
        elif type(path) == int or path is None:
            self.vel_linear_x = 0.0
            self.vel_angular_z = 0.0
            return

        if (self.count >20):

            if not self.angle_relation_computed:

                self.velocity_linear_x = 0.0

                # Extracting Car angle (Img) from car_InitLoc and car_FinalLoc after moving forward (50 iters)
                self.bot_angle, _= self.angle_n_dist(self.init_loc, bot_loc)
                self.bot_angle_init = self.bot_angle
                # Finding relation coeffiecient between car_angle (Image <-> Simulation)
                self.bot_angle_rel = self.bot_angle_s - self.bot_angle
                self.angle_relation_computed = True

            else:
                # [For noob luqman] : Extracting Car angle [From Simulation angle & S-I Relation]
                self.bot_angle = self.bot_angle_s - self.bot_angle_rel

                if (config.debug and config.debug_motionplanning):
                    print("\n\nCar angle (Image From Relation) = {} I-S Relation {} Car angle (Simulation) = {}".format(self.bot_angle,self.bot_angle_rel,self.bot_angle_s))
                    print("Car angle_Initial (Image) = ",self.bot_angle_init)
                    print("Car loc {}".format(bot_loc))

                # [NEW]: GotoGoal now takes only 2 argunment
                #        2 Arguemnts are removed velovity and velocity publisher
                # Traversing through found path to reach goal. If path planning
                # failed, stop safely instead of crashing/indexing an int.
                if type(path) != int and path is not None and len(path) >= 2:
                    self.go_to_goal(bot_loc,path)
                else:
                    # No route: stop command output, not just odom display vars.
                    self.vel_linear_x = 0.0
                    self.vel_angular_z = 0.0
                    self.actual_speed = 0.0
                    self.actual_angle = 0.0


        else:

            # [NEW]: Only proceed if bot lies within the road network
            if all(bot_loc_wrt_rdnetwrk>0):
                # If bot initial location not already taken
                if not self.pt_i_taken:
                    # Set init_loc = Current bot location
                    self.init_loc = bot_loc
                    self.pt_i_taken = True
                    
                # Keep moving forward for 20 iterations(count)
                self.velocity_linear_x = 1.0

                self.count+=1
