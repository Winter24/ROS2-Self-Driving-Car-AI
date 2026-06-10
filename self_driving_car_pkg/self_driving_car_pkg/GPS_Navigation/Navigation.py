'''
> Purpose :
Node to perform the actual (worthy of your time) task of maze solving ;) 
- Robot velocity interface
- Upper Video camera as well

> Usage :
You need to write below command in terminal where your pacakge is sourced
- ros2 run maze_bot maze_solver

Note : Name of the node is actually name of executable file described in setup.py file of our package and not the name of python file

> Inputs:
This node is subscribing video feed from (Satellite or DroneCam)

> Outputs:
This node publishes on topic "/cmd_vel" , the required velocity ( linear and angular ) to move the robot

Author :
Haider Abbasi

Date :
18/03/22
'''


import cv2
import numpy as np

from .bot_localization import bot_localizer
from .bot_mapping import bot_mapper
from .bot_pathplanning import bot_pathplanner
from .bot_motionplanning import bot_motionplanner

# importing utility functions for taking destination from user
from .utilities import Debugging,click_event,find_point_in_FOR
import sys
from . import config
# functionality to provide on device prompt to user to select destination
from .utilities import disp_on_mydev
# motionplanning (Visualization) Imports
from .utilities_disp import disp_SatNav
from .manual_map import load_manual_map, nearest_node_id, shortest_node_path, path_road_xy, path_sat_xy


def _path_pt_to_satview(pt, bot_localizer, crp_amt=0):
    """Convert path point from cropped road-network frame back to sat_view pixel."""
    transform_arr = bot_localizer.transform_arr
    rot_mat = bot_localizer.rot_mat_rev

    # Path points are generated on bot_mapper.maze, which is the rotated
    # occupancy grid cropped by crp_amt on every side. Add that crop back
    # before applying the inverse rotation/translation to SatView.
    pt_array = np.array([pt[0] + crp_amt, pt[1] + crp_amt])
    rot_center = (rot_mat @ pt_array.T).T

    st_col = transform_arr[0]
    st_row = transform_arr[1]
    rot_cols = transform_arr[2]
    rot_rows = transform_arr[3]

    rot_center[0] = rot_center[0] + (rot_cols * (rot_center[0] < 0)) + st_col
    rot_center[1] = rot_center[1] + (rot_rows * (rot_center[1] < 0)) + st_row
    return (int(rot_center[0]), int(rot_center[1]))


def _draw_path_on_satview(frame_disp, path, bot_localizer, crp_amt=0):
    """Draw the actual planned path_to_goal directly on SatView live."""
    if type(path) == int or path is None or len(path) < 2:
        cv2.putText(frame_disp, "GPS path: unavailable/too short", (16, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return

    pts = []
    for pt in path:
        try:
            p = _path_pt_to_satview(pt, bot_localizer, crp_amt)
            if 0 <= p[0] < frame_disp.shape[1] and 0 <= p[1] < frame_disp.shape[0]:
                pts.append(p)
        except Exception:
            pass

    if len(pts) < 2:
        cv2.putText(frame_disp, "GPS path: transform failed", (16, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        print("[GPS Path Debug] path exists but transform produced <2 visible points. len(path)=", len(path))
        return

    for i in range(len(pts) - 1):
        cv2.line(frame_disp, pts[i], pts[i + 1], (0, 0, 255), 5)
        cv2.line(frame_disp, pts[i], pts[i + 1], (255, 255, 255), 2)

    cv2.circle(frame_disp, pts[0], 8, (255, 0, 0), -1)       # start blue
    cv2.circle(frame_disp, pts[-1], 10, (0, 255, 0), -1)     # destination green
    cv2.putText(frame_disp, "GPS path", (16, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



def _draw_manual_map_on_satview(frame_disp, manual_map, node_path_ids=None):
    """Draw saved waypoint graph and selected node route on SatView."""
    if not manual_map:
        return
    nodes = {n["id"]: n for n in manual_map.get("nodes", [])}
    # saved graph edges
    for e in manual_map.get("edges", []):
        a = nodes.get(e.get("from"))
        b = nodes.get(e.get("to"))
        if not a or not b or "sat_xy" not in a or "sat_xy" not in b:
            continue
        p1 = tuple(map(int, a["sat_xy"]))
        p2 = tuple(map(int, b["sat_xy"]))
        cv2.line(frame_disp, p1, p2, (60, 60, 60), 2)
    # saved nodes
    for n in manual_map.get("nodes", []):
        if "sat_xy" not in n:
            continue
        p = tuple(map(int, n["sat_xy"]))
        cv2.circle(frame_disp, p, 5, (0, 255, 255), -1)
        cv2.putText(frame_disp, str(n["id"]), (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    # active route on graph
    if node_path_ids and len(node_path_ids) >= 2:
        for i in range(len(node_path_ids)-1):
            a = nodes.get(node_path_ids[i])
            b = nodes.get(node_path_ids[i+1])
            if not a or not b:
                continue
            p1 = tuple(map(int, a["sat_xy"]))
            p2 = tuple(map(int, b["sat_xy"]))
            cv2.line(frame_disp, p1, p2, (0, 0, 255), 5)
            cv2.line(frame_disp, p1, p2, (255, 255, 255), 2)
        sp = tuple(map(int, nodes[node_path_ids[0]]["sat_xy"]))
        ep = tuple(map(int, nodes[node_path_ids[-1]]["sat_xy"]))
        cv2.circle(frame_disp, sp, 9, (255, 0, 0), -1)
        cv2.circle(frame_disp, ep, 11, (0, 255, 0), -1)
        cv2.putText(frame_disp, "Manual GPS path", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

class Navigator():

    def __init__(self):
        
        # Creating objects for each stage of the robot navigation
        self.bot_localizer = bot_localizer()
        self.bot_mapper = bot_mapper()
        self.bot_pathplanner = bot_pathplanner()
        self.bot_motionplanner = bot_motionplanner()

        self.debugging = Debugging()

        # [NEW]: Boolean to determine if we are taking destination from user or not
        self.accquiring_destination = True
        # [NEW]: Container to store destination selected by User
        self.destination = []

        # [NEW]: Displays the satellite view inside the screen of a device
        self.device_view = []
        # [NEW]: Screen (start_x,start_y) for passing satellite view to display
        self.screen_x = 0
        self.screen_y = 0

        # Manual waypoint graph route state. This replaces image skeleton/graphify.
        self.manual_map = None
        self.manual_node_path = []
        self.manual_path = []
        self.manual_route_ready = False
        self.accquiring_start = True
        self.manual_start = None
        self.manual_start_sat = None
        self.manual_start_node_id = None
        self.destination_sat = None
        self.destination_node_id = None



    def _ensure_manual_map_loaded(self):
        if self.manual_map is None:
            self.manual_map = load_manual_map()
            print("[Manual GPS Debug] loaded manual waypoint map: nodes={}, edges={}".format(
                len(self.manual_map.get("nodes", [])), len(self.manual_map.get("edges", []))))

    def _pick_manual_start(self, sat_view):
        """Ask user to click the waypoint node representing current car start."""
        self._ensure_manual_map_loaded()
        if len(self.manual_map.get("nodes", [])) == 0:
            print("[Manual GPS Debug] ERROR: no waypoint nodes. Run gps_waypoint_editor first.")
            return False

        config.destination = []
        view = sat_view.copy()
        _draw_manual_map_on_satview(view, self.manual_map, None)
        cv2.putText(view, "Click START waypoint (current car position)", (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.namedWindow("Select START waypoint", cv2.WINDOW_NORMAL)
        cv2.imshow("Select START waypoint", view)
        cv2.setMouseCallback("Select START waypoint", click_event, {
            "sat_w": sat_view.shape[1], "sat_h": sat_view.shape[0], "view": view,
            "window_name": "Select START waypoint",
            "marker_text": "START",
        })
        while config.destination == []:
            cv2.waitKey(1)
        click_sat = tuple(config.destination)
        start_id, d = nearest_node_id(self.manual_map, click_sat, key="sat_xy")
        nodes = {n["id"]: n for n in self.manual_map.get("nodes", [])}
        n = nodes.get(start_id)
        if n is None or "road_xy" not in n:
            print("[Manual GPS Debug] ERROR: selected start node invalid", start_id)
            return False
        self.manual_start_node_id = start_id
        self.manual_start = tuple(map(int, n["road_xy"]))
        self.manual_start_sat = tuple(map(int, n.get("sat_xy", click_sat)))

        # Use the user-selected START waypoint as the initial localization prior.
        # The automatic satellite detector is unreliable in this world and can
        # lock onto a wrong object. This does NOT freeze the car forever: it only
        # seeds the tracker at the real car/start position, then future frames
        # search near this prior and update normally.
        self.bot_localizer.last_car_sat_xy = self.manual_start_sat
        self.bot_localizer.loc_car = self.manual_start
        try:
            self.bot_localizer.loc_car_wrt_rdntwork = np.array([
                self.manual_start_sat[0] - self.bot_localizer.orig_X,
                self.manual_start_sat[1] - self.bot_localizer.orig_Y,
            ])
        except Exception:
            pass

        cv2.destroyWindow("Select START waypoint")
        print("[Manual GPS Debug] selected START click_sat={} -> node {} sat_xy={} road_xy={} d={:.1f}".format(
            click_sat, start_id, n.get("sat_xy"), self.manual_start, d))
        print("[GPS Loc Debug] manual START seeded as LIVE CAR sat_xy={} road_xy={}".format(
            self.manual_start_sat, self.manual_start))
        return True

    # [NEW]: Adding Car_dash view to the mix to see both the self drive and Sat-Nav at the same time
    def navigate_to_home(self,sat_view,bot_view):
        """ Performs Visual-Navigation (like GPS) by utilizing video-feed received from satellite.

        Args:
            sat_view (numpy_nd_array): Visual feed (curr_frame) from the satellite
            bot_view (numpy_nd_array): Prius dash-cam view
        """        
        self.debugging.setDebugParameters()

        # Creating frame to display current robot state to user        
        frame_disp = sat_view.copy()
        
        # Manual GPS mode intentionally does NOT localize the car from image
        # difference/background subtraction anymore. That approach is unstable
        # with water/shadows/other moving objects. Control uses /odom x/y/yaw.

        # First ask user to select the current car/start waypoint from the saved manual map.
        if self.accquiring_start:
            if not self._pick_manual_start(sat_view):
                cv2.imshow("SatView (Live)", frame_disp)
                cv2.waitKey(1)
                return
            self.accquiring_start = False
            # After selecting START, immediately prompt for DESTINATION on the
            # next pass instead of keeping the app stuck on the start picker.
            self.accquiring_destination = True
            cv2.destroyWindow("Select START waypoint")

        # (NEW): Acquiring Destination from the User
        if self.accquiring_destination:
            # Clean destination picker: display the satellite map directly instead
            # of embedding it inside a tablet image. Click coordinates now match
            # sat_view pixel coordinates exactly.
            config.destination = []
            self.device_view = sat_view.copy()
            _draw_manual_map_on_satview(self.device_view, self.manual_map, None)
            cv2.putText(self.device_view, "Click DESTINATION", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            self.screen_x = 0
            self.screen_y = 0
            cv2.namedWindow("Mark your destination!!!",cv2.WINDOW_NORMAL)
            cv2.imshow("Mark your destination!!!",self.device_view)
            cv2.setMouseCallback("Mark your destination!!!", click_event, {
                "sat_w": sat_view.shape[1],
                "sat_h": sat_view.shape[0],
                "view": self.device_view,
                "window_name": "Mark your destination!!!",
                "marker_text": "DEST",
            })
            while(self.destination==[]):
                self.destination = config.destination
                cv2.waitKey(1)
            if self.destination!=[]:
                self.destination_sat = tuple(map(int, self.destination))
                print("[GPS Click Debug] selected DEST on sat_view = ", self.destination_sat)
                print("[GPS Click Debug] sat_view shape h,w       = ", sat_view.shape[0], sat_view.shape[1])

                cv2.destroyWindow("Mark your destination!!!")
                self.accquiring_destination = False
                cv2.namedWindow("SatView (Live)",cv2.WINDOW_NORMAL)
                print("Destination selected by user (sat_xy) = ", self.destination_sat)
            else:
                print("Destination not specified.... Exiting!!!")
                sys.exit()

        # [Stage 2/3: Manual waypoint graph planning]
        # Route planning uses only the saved manual waypoint graph. The selected
        # START and DEST clicks are snapped to nearest waypoint nodes by sat_xy.
        if not self.manual_route_ready:
            try:
                self.manual_map = load_manual_map()
                if len(self.manual_map.get("nodes", [])) < 2:
                    print("[Manual GPS Debug] ERROR: manual map has <2 nodes. Run gps_waypoint_editor first.")
                    path = -1
                else:
                    start_id = self.manual_start_node_id
                    if start_id is None:
                        print("[Manual GPS Debug] ERROR: start waypoint has not been selected")
                        path = -1
                    else:
                        end_id, end_d = nearest_node_id(self.manual_map, self.destination_sat, key="sat_xy")
                        self.destination_node_id = end_id
                        self.manual_node_path = shortest_node_path(self.manual_map, start_id, end_id, key="sat_xy")
                        if len(self.manual_node_path) < 2:
                            print("[Manual GPS Debug] ERROR: no graph route from node {} to node {}".format(start_id, end_id))
                            path = -1
                        else:
                            # sat path is used for stable /odom controller; road path is
                            # kept only for backwards-compatible debug prints.
                            self.manual_path = path_sat_xy(self.manual_map, self.manual_node_path)
                            self.manual_route_ready = True
                            path = self.manual_path
                            print("[Manual GPS Debug] start node {} sat_xy={}".format(start_id, self.manual_start_sat))
                            print("[Manual GPS Debug] dest click sat_xy={} -> node {} d={:.1f}".format(self.destination_sat, end_id, end_d))
                            print("[Manual GPS Debug] node path =", self.manual_node_path)
                            print("[Manual GPS Debug] sat path length =", len(path))
                            ok = self.bot_motionplanner.configure_odom_path_from_sat(path)
                            if not ok:
                                print("[GPS Odom Debug] WARNING: could not configure odom path yet")
            except Exception as e:
                print("[Manual GPS Debug] ERROR loading/planning manual map:", e)
                path = -1
        else:
            path = self.manual_path

        if self.manual_route_ready and len(self.bot_motionplanner.odom_path) == 0:
            ok = self.bot_motionplanner.configure_odom_path_from_sat(self.manual_path)
            if not ok:
                print("[GPS Odom Debug] waiting to configure odom path")

        # [Stage 4: Odom MotionPlanning]
        # Follow the selected manual path using /odom x/y/yaw. No image-based
        # car localization is used for control.
        print("[GPS Path Debug] node_path =", self.manual_node_path, " path_len=",
              (len(path) if type(path) != int and path is not None else path))
        self.bot_motionplanner.nav_path_odom()
        # Draw manual graph + selected route directly in SatView coordinates.
        _draw_manual_map_on_satview(frame_disp, self.manual_map, self.manual_node_path)

        # Draw current odom-estimated car position projected back onto SatView.
        odom_sat = self.bot_motionplanner.world_to_sat((self.bot_motionplanner.pose_x, self.bot_motionplanner.pose_y))
        if odom_sat is not None:
            cv2.circle(frame_disp, odom_sat, 12, (255, 0, 255), 3)
            cv2.putText(frame_disp, "ODOM CAR", (odom_sat[0] + 12, odom_sat[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

            # Camera-dominant steering can drive slightly beside the GPS odom
            # path, so world-waypoint distance alone may miss the final target.
            # Stop when the projected odom marker reaches or passes the selected
            # destination on the SatView route.
            if self.destination_sat is not None and self.manual_path and len(self.manual_path) >= 2:
                dest = tuple(map(float, self.manual_path[-1]))
                prev = tuple(map(float, self.manual_path[-2]))
                cur = tuple(map(float, odom_sat))
                dx = dest[0] - prev[0]
                dy = dest[1] - prev[1]
                seg_len2 = dx * dx + dy * dy
                along_t = 0.0
                if seg_len2 > 1e-9:
                    along_t = ((cur[0] - prev[0]) * dx + (cur[1] - prev[1]) * dy) / seg_len2
                dest_dist = float(np.hypot(cur[0] - dest[0], cur[1] - dest[1]))
                if self.bot_motionplanner.goal_not_reached_flag and (dest_dist <= 15.0 or along_t >= 1.0):
                    self.bot_motionplanner.goal_not_reached_flag = False
                    self.bot_motionplanner.vel_linear_x = 0.0
                    self.bot_motionplanner.vel_angular_z = 0.0
                    print("[GPS Stop Debug] destination reached/passed on SatView odom_sat={} dest={} dist={:.1f}px along={:.2f}".format(
                        odom_sat, tuple(map(int, dest)), dest_dist, along_t))

        # Simple HUD and bot camera overlay; avoid old skeleton/SatNav overlay.
        curr_speed = self.bot_motionplanner.actual_speed
        curr_angle = self.bot_motionplanner.actual_angle
        cv2.rectangle(frame_disp, (8, 4), (330, 32), (0,0,0), -1)
        cv2.putText(frame_disp, "GPS Odom | v {:.2f} | w {:.2f} | wp {}/{}".format(
                    self.bot_motionplanner.vel_linear_x, self.bot_motionplanner.vel_angular_z,
                    self.bot_motionplanner.odom_path_iter, max(0, len(self.bot_motionplanner.odom_path)-1)),
                    (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
        if bot_view is not None and hasattr(bot_view, 'shape'):
            h = min(260, bot_view.shape[0])
            w = min(450, bot_view.shape[1])
            view = cv2.resize(bot_view, (w, h))
            y0 = max(0, frame_disp.shape[0] - h - 10)
            x0 = 10
            frame_disp[y0:y0+h, x0:x0+w] = view
            cv2.rectangle(frame_disp, (x0, y0), (x0+w, y0+h), (0,255,255), 3)
            cv2.putText(frame_disp, "Bot View", (x0+6, y0+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        # [NEW]: Displaying Satellite Navigation directly on the map canvas.
        # Do not wrap the live view inside the tablet image; it makes the UI too
        # small and hard to read. The tablet image is only used for destination selection.
        cv2.imshow("SatView (Live)", frame_disp)
        cv2.waitKey(1)
