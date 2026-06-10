'''
> Purpose :
Module to perform mapping to convert [(top down) maze view ==> traversable graph.]

> Usage :
You can perform mapping by
1) Importing the class (bot_mapper)
2) Creating its object
3) Accessing the object's function of (graphify). 
E.g ( self.bot_mapper.graphify(self.bot_localizer.maze_og) )


> Inputs:
1) Occupancy Grid from localization stage

> Outputs:
1) self.Graph.graph       => Generated graph from provided maze occupancy grid
2) self.maze              => Image displaying only pathways in maze

Author :
Haider Abbasi

Date :
6/04/22
'''
import cv2
import numpy as np
import os

from . import config
# Importing utility functions to help in estimating start and end for graph
from .utilities import closest_node,get_centroid
draw_intrstpts = True
debug_mapping = False

def _gps_viz(name, img):
    if not getattr(config, "visualize_pipeline", False):
        return
    try:
        os.makedirs(getattr(config, "visualize_pipeline_dir", "/tmp/gps_nav_debug"), exist_ok=True)
        cv2.imwrite(os.path.join(config.visualize_pipeline_dir, name + ".png"), img)
        cv2.namedWindow("GPS DEBUG - " + name, cv2.WINDOW_NORMAL)
        cv2.imshow("GPS DEBUG - " + name, img)
        cv2.waitKey(1)
    except Exception as e:
        print("[GPS Viz Debug] failed", name, e)



def thinning(binary_img):
    """Return skeletonized binary image; fallback when cv2.ximgproc.thinning is unavailable."""
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary_img)

    img = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY)[1].copy()
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

# Creating Graph Class to store IP and their connected paths
class Graph():

    def __init__(self):
        # Dictionary to store graph
        self.graph = {}
        # Placeholder for start and end of graph
        self.start = 0
        self.end = 0

    # function to add new vertex to graph
    # if neighbor == None  Just add vertex
    #      Otherwise add connection
    def add_vertex(self,vertex,neighbor= None,case = None, cost = None):
        
        # Ensure the vertex exists first.
        if vertex not in self.graph.keys():
            self.graph[vertex] = {}
            self.graph[vertex]["case"] = case

        # If neighbor == None, just add/update the vertex case. Do NOT create a
        # None neighbor, because path planners iterate graph[u] and expect every
        # non-"case" key to be a valid vertex.
        if neighbor is None:
            self.graph[vertex]["case"] = case
            return

        # If neighbor is present ==> Add connection
        self.graph[vertex][neighbor] = {}
        self.graph[vertex][neighbor]["case"] = case
        self.graph[vertex][neighbor]["cost"] = cost

    # Function to display complete graph
    def displaygraph(self):
        for key,value in self.graph.items():
            print("key {} has value {} ".format(key,value))

# Bot_Mapper Class for performing Stage 2 (mapping) of robot navigation
class bot_mapper():

    def __init__(self):

        # State Variables
        self.graphified = False

        # Cropping control for removing maze boundary
        self.crp_amt = 5

        # Creating a graph object for storing Maze
        self.Graph = Graph()

        # State variables to define the connection status of each vertex
        self.connected_left = False
        self.connected_upleft = False
        self.connected_up = False
        self.connected_upright = False
        # Maze (Colored) for displaying connection between nodes
        self.maze_connect = []

        self.maze_interestPts = []

        # Maze (One-Pass) Input
        self.maze = 0

        # [New]: container to store found dcsn_pts [Intersection & T-Junc]
        self.maze_dcsn_pts = []


    # Display connection between nodes with a colored line
    def display_connected_nodes(self,curr_node,neighbor_node,case="Unkown",color=(0,0,255)):
        curr_pixel = (curr_node[1],curr_node[0])
        neighbor_pixel = (neighbor_node[1],neighbor_node[0])
        #self.maze_connect= cv2.circle(self.maze_connect, curr_pixel, 5, (255,0,0))
        #self.maze_connect= cv2.circle(self.maze_connect, neighbor_pixel, 5, (255,0,0))
        if debug_mapping:
            print("----------------------) CONNECTED >> {} << ".format(case))
        self.maze_connect = cv2.line(self.maze_connect,curr_pixel,neighbor_pixel,color,1)
        if config.debug and config.debug_mapping:
            cv2.imshow("Nodes Conected", self.maze_connect)
        if debug_mapping:
            cv2.waitKey(0)                    
            self.maze_connect = cv2.line(self.maze_connect,curr_pixel,neighbor_pixel,(255,255,255),1)

    # Connect curr_node to its neighbors in immediate [left -> top-right] region 
    def connect_neighbors(self,maze,node_row,node_col,case,step_l = 1,step_up = 0,totl_cnncted = 0):
        
        curr_node = (node_row,node_col)

        # Boundary guard for non-original worlds whose skeleton can touch image borders.
        lookup_row = node_row - step_up
        lookup_col = node_col - step_l
        in_bounds = (0 <= lookup_row < maze.shape[0]) and (0 <= lookup_col < maze.shape[1])

        # Check if there is a path around our node        
        if (in_bounds and maze[lookup_row][lookup_col]>0):
            # There is a path ==> Look for neighbor node to connect
            neighbor_node = (node_row-step_up,node_col-step_l)
            # If potential_neighbor_node is actually a key in graph                
            if neighbor_node in self.Graph.graph.keys():
                neighbor_case = self.Graph.graph[neighbor_node]["case"]
                cost = max(abs(step_l),abs(step_up))
                totl_cnncted +=1

                self.Graph.add_vertex(curr_node,neighbor_node,neighbor_case,cost)
                self.Graph.add_vertex(neighbor_node,curr_node,case,cost)
                if debug_mapping:
                    print("\nConnected {} to {} with Case [step_l,step_up] = [ {} , {} ] & Cost -> {}".format(curr_node,neighbor_node,step_l,step_up,cost))

                # Vertex <-Connected-> Neighbor ===) Cycle through Next Possible Routes [topleft,top,top_right]
                if not self.connected_left:
                    self.display_connected_nodes(curr_node, neighbor_node,"LEFT",(0,0,255))
                    # Vertex has connected to its left neighbor.                    
                    self.connected_left = True
                    # Check up-Left route now
                    step_l = 1
                    step_up = 1
                    self.connect_neighbors(maze, node_row, node_col, case,step_l,step_up,totl_cnncted)
                if not self.connected_upleft:
                    self.display_connected_nodes(curr_node, neighbor_node,"UPLEFT",(0,128,255))
                    # Vertex has connected to its up-left neighbor.
                    self.connected_upleft = True
                    # Check top route now
                    step_l  = 0
                    step_up = 1
                    self.connect_neighbors(maze, node_row, node_col, case,step_l,step_up,totl_cnncted)
                if not self.connected_up:
                    self.display_connected_nodes(curr_node, neighbor_node,"UP",(0,255,0))
                    # Vertex has connected to its up neighbor.
                    self.connected_up = True
                    # Check top-right route now
                    step_l  = -1
                    step_up = 1
                    self.connect_neighbors(maze, node_row, node_col, case,step_l,step_up,totl_cnncted)
                if not self.connected_upright:
                    self.display_connected_nodes(curr_node, neighbor_node,"UPRIGHT",(255,0,0))
                    # Vertex has connected to its up-right neighbor.
                    self.connected_upright = True

            # Still searching for node to connect in a respective direction
            if not self.connected_upright:
                if not self.connected_left:
                    # Look a little more left, You'll find it ;)
                    step_l +=1
                elif not self.connected_upleft:
                    # Look a little more (diagnolly) upleft, You'll find it ;)
                    step_l+=1
                    step_up+=1
                elif not self.connected_up:
                    # Look a little more up, You'll find it ;)
                    step_up+=1
                elif not self.connected_upright:
                    # Look a little more upright, You'll find it ;)
                    step_l-=1
                    step_up+=1
                self.connect_neighbors(maze, node_row, node_col, case,step_l,step_up,totl_cnncted)
        else:
            # No path in the direction you are looking, Cycle to next direction
            if not self.connected_left:
                # Basically there is a wall on left so just start looking up lft:)
                self.connected_left = True
                # Looking upleft now
                step_l = 1
                step_up = 1
                self.connect_neighbors(maze, node_row, node_col, case,step_l,step_up,totl_cnncted)
            elif not self.connected_upleft:
                # Basically there is a wall up lft so just start looking up :)
                self.connected_upleft = True
                step_l = 0
                step_up = 1
                self.connect_neighbors(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)
                

            elif not self.connected_up:
                # Basically there is a wall above so just start looking up-right :)
                self.connected_up = True
                step_l = -1
                step_up = 1
                self.connect_neighbors(maze, node_row, node_col, case, step_l, step_up, totl_cnncted)

            elif not self.connected_upright:
                # Basically there is a wall above so just start looking up-right :)
                self.connected_upright = True
                step_l = 0
                step_up = 0                
                return     
    
    # function to draw a triangle around a point    
    @staticmethod
    def triangle(image,ctr_pt,radius,colour=(0,255,255),thickness=2):
        # Polygon corner points coordinates
        pts = np.array( [ [ctr_pt[0]        , ctr_pt[1]-radius]  , 
                          [ctr_pt[0]-radius , ctr_pt[1]+radius]  ,
                          [ctr_pt[0]+radius , ctr_pt[1]+radius]   
                        ] 
                        ,np.int32
                      )
        
        pts = pts.reshape((-1, 1, 2))
        
        image = cv2.polylines(image,[pts],True,colour,thickness)
        return image
    
    # function to get surrounding pixels intensities for any point
    @staticmethod
    def get_surround_pixel_intensities(maze,curr_row,curr_col):

        # binary thrsholding and setting (+ values ==> 1 
        #                                 - values ==> 0)
        maze = cv2.threshold(maze, 1, 1, cv2.THRESH_BINARY)[1]

        rows = maze.shape[0]
        cols = maze.shape[1]

        # State variables , If our point is at a boundary condition
        top_row = False
        btm_row = False
        lft_col = False
        rgt_col = False

        # Checking if there is a boundary condition
        if (curr_row==0):
            # Top row => Row above not accesible
            top_row = True
        if (curr_row == (rows-1)):
            # Bottom row ==> Row below not accesible
            btm_row = True
        if (curr_col == 0):
            # Left col ==> Col to the left not accesible
            lft_col = True
        if (curr_col == (cols-1)):
            # Right col ==> Col to the right not accesible
            rgt_col = True

        # Extracting surround pixel intensities and Addressing boundary conditions (if present)
        if (top_row or lft_col):
            top_left = 0
        else:
            top_left = maze[curr_row-1][curr_col-1]
        if( top_row or rgt_col ):
            top_rgt = 0
        else:
            top_rgt = maze[curr_row-1][curr_col+1]

        if( btm_row or lft_col ):
            btm_left = 0
        else:
            btm_left = maze[curr_row+1][curr_col-1]

        if( btm_row or rgt_col ):
            btm_rgt = 0
        else:
            btm_rgt = maze[curr_row+1][curr_col+1]
        
        # If the point we are at is anywhere on the top row, Then
        #             ===> Its top pixel is definitely not accesible
        if (top_row):
            top = 0
        else:
            top = maze[curr_row-1][curr_col]
        if (rgt_col):
            rgt = 0
        else:
            rgt = maze[curr_row][curr_col+1]
        
        if (btm_row):
            btm = 0
        else:
            btm = maze[curr_row+1][curr_col]

        if (lft_col):
            lft = 0
        else:
            lft = maze[curr_row][curr_col-1]

        no_of_pathways = ( top_left + top      + top_rgt  +
                           lft      + 0        + rgt      + 
                           btm_left + btm      + btm_rgt        
                         )
        if ( (no_of_pathways>2) and (debug_mapping) ):  
            print("  [ top_left , top      , top_rgt  ,lft    , rgt      , btm_left , btm      , btm_rgt   ] \n [ ",str(top_left)," , ",str(top)," , ",str(top_rgt)," ,\n   ",str(lft)," , ","-"," , ",str(rgt)," ,\n   ",str(btm_left)," , ",str(btm)," , ",str(btm_rgt)," ] ")
            print("\nno_of_pathways [row,col]= [ ",curr_row," , ",curr_col," ] ",no_of_pathways) 

        return top_left,top,top_rgt,rgt,btm_rgt,btm,btm_left,lft,no_of_pathways
    # Reset state parameters of each vertex connection
    def reset_connct_paramtrs(self):
        # Reseting member variables to False initially when looking for nodes to connect
        self.connected_left = False
        self.connected_upleft = False
        self.connected_up = False
        self.connected_upright = False

    def one_pass(self,maze,start_loc=[],destination=[]):

        # Remove previously found nodes
        self.Graph.graph.clear()

        # Initalizing Maze_connect with Colored Maze
        self.maze_connect = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("Nodes Conected",cv2.WINDOW_FREERATIO)

        # Initialize counts of Ip's
        turns = 0
        junc_3 = 0
        junc_4 = 0

        # [NEW]: Converting maze to Colored for Identifying discovered Interest Points
        if not draw_intrstpts:
            maze_bgr = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
        else:
            maze_bgr = np.zeros((maze.shape[0],maze.shape[1],3),np.uint8)

        # [NEW]: Drawing an image Indicating detected decision points
        self.maze_dcsn_pts = np.zeros_like(maze)
        # Creating a window to display Detected Interest Points
        cv2.namedWindow("Maze (Interest Points)",cv2.WINDOW_FREERATIO)
        rows = maze.shape[0]
        cols = maze.shape[1]

        # Looping over each pixel from left to right ==> bottom to top
        for row in range(rows):
            for col in range(cols):

                if (maze[row][col]==255):
                    if debug_mapping:
                        # Re-Initalizing Maze_connect with Colored Maze
                        self.maze_connect = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
                    # Probable IP => Find Surrounding Pixel Intensities
                    top_left,top,top_rgt,rgt,btm_rgt,btm,btm_left,lft, paths = self.get_surround_pixel_intensities(maze.copy(),row,col)

                    # [NEW]: Adjusting for case when start and destination have already been provided
                    if ( ( (start_loc == (col,row) ) or (destination == (col,row)) ) or
                         ( (start_loc==[]) and 
                           ( (row==0) or (row == (rows-1)) or (col==0) or (col == (cols-1)) )
                         ) 
                       ):
                        # [NEW]: Adding Case when start location have been provided 
                        if ( (start_loc == (col,row)) or ( (start_loc==[]) and (row == 0) ) ):
                            # Start
                            maze_bgr[row][col] = (0,128,255)
                            # [NEW]: indicate start by circle (if draw_interest points)
                            if draw_intrstpts:
                                maze_bgr= cv2.circle(maze_bgr, (col,row), 15, (0,140,255),-1)
                            if config.debug and config.debug_mapping:
                                cv2.imshow("Maze (Interest Points)",maze_bgr)
                            # Adding [Found vertex to graph & maze entry to graph-start]
                            self.Graph.add_vertex((row,col),case="_Start_")
                            self.Graph.start = (row,col)

                            # [NEW]: Connecting Start to its neighbor as it maybe anwhere on Map (if-any)
                            self.reset_connct_paramtrs()
                            self.connect_neighbors(maze, row, col, "_Start_")

                        # [NEW]: Case when destination location have been provided 
                        elif ( (destination == (col,row)) or (start_loc==[]) ):
                            # End (MAze Exit)
                            maze_bgr[row][col] = (0,255,0)

                            # [NEW]: indicate end by green circle (if draw_interest points)
                            if draw_intrstpts:
                                maze_bgr= cv2.circle(maze_bgr, (col,row), 15, (0,255,0),-1)
                            if config.debug and config.debug_mapping:
                                cv2.imshow("Maze (Interest Points)",maze_bgr)
                            # Adding [Found vertex to graph & maze exit to graph-end]
                            self.Graph.add_vertex((row,col),case="_End_")
                            self.Graph.end = (row,col)
                            # Connecting vertex to its neighbor (if-any)
                            self.reset_connct_paramtrs()
                            self.connect_neighbors(maze, row, col, "_End_")

                    # Check if it is a (Dead End)
                    elif (paths==1):
                        crop = maze[row-1:row+2,col-1:col+2]
                        if debug_mapping:
                            print(" ** [Dead End] ** \n" ,crop)
                        maze_bgr[row][col] = (0,0,255)# Red color
                        if draw_intrstpts:
                            maze_bgr= cv2.circle(maze_bgr, (col,row), 10, (0,0,255),4)
                        if config.debug and config.debug_mapping:
                            cv2.imshow("Maze (Interest Points)",maze_bgr)
                        # Adding [Found vertex to graph]
                        self.Graph.add_vertex((row,col),case = "_DeadEnd_")
                        # Connecting vertex to its neighbor (if-any)
                        self.reset_connct_paramtrs()
                        self.connect_neighbors(maze, row, col, "_DeadEnd_")

                    # Check if it is either a *Turn* or just an ordinary path
                    elif (paths==2):
                        crop = maze[row-1:row+2,col-1:col+2]
                        nzero_loc = np.nonzero(crop > 0)
                        nzero_ptA = (nzero_loc[0][0],nzero_loc[1][0])
                        nzero_ptB = (nzero_loc[0][2],nzero_loc[1][2])
                        if not ( ( (2 - nzero_ptA[0])==nzero_ptB[0] ) and 
                                    ( (2 - nzero_ptA[1])==nzero_ptB[1] )     ):
                            #maze_bgr[row][col] = (255,0,0)
                            #if draw_intrstpts:
                                #maze_bgr= cv2.circle(maze_bgr, (col,row), 10, (255,0,0),2)
                            if config.debug and config.debug_mapping:
                                cv2.imshow("Maze (Interest Points)",maze_bgr)
                            # Adding [Found vertex to graph]
                            self.Graph.add_vertex((row,col),case = "_Turn_")
                            # Connecting vertex to its neighbor (if-any)
                            self.reset_connct_paramtrs()
                            self.connect_neighbors(maze, row, col, "_Turn_")
                            turns+=1
                    # Check if it is either a *3-Junc* or a *4-Junc*
                    elif (paths>2):
                        if (paths ==3):
                            maze_bgr[row][col] = (255,244,128)
                            # [NEW]: Identify T-Junc in maze_dcsn_pts by drawing
                            self.maze_dcsn_pts[row][col] = 255
                            # [NEW]: Turn off to avoid drawing before confirmation
                            if draw_intrstpts:
                                pass
                                #maze_bgr = self.triangle(maze_bgr, (col,row), 10,(144,140,255),4)
                            if config.debug and config.debug_mapping:
                                cv2.imshow("Maze (Interest Points)",maze_bgr)
                            # Adding [Found vertex to graph]
                            self.Graph.add_vertex((row,col),case = "_3-Junc_")
                            # Connecting vertex to its neighbor (if-any)
                            self.reset_connct_paramtrs()
                            self.connect_neighbors(maze, row, col, "_3-Junc_")
                            junc_3+=1                                   
                        else:
                            maze_bgr[row][col] = (128,0,128)
                            # [NEW]: Identify Intersection in maze_dcsn_pts by drawing
                            self.maze_dcsn_pts[row][col] = 255

                            # [NEW]: Turn off to avoid drawing before confirmation
                            if draw_intrstpts:
                                pass
                                #cv2.rectangle(maze_bgr,(col-10,row-10) , (col+10,row+10), (255,215,0),4)
                            if config.debug and config.debug_mapping:
                                cv2.imshow("Maze (Interest Points)",maze_bgr)
                            # Adding [Found vertex to graph]
                            self.Graph.add_vertex((row,col),case = "_4-Junc_")
                            # Connecting vertex to its neighbor (if-any)
                            self.reset_connct_paramtrs()
                            self.connect_neighbors(maze, row, col, "_4-Junc_")
                            junc_4+=1
        self.maze_interestPts = maze_bgr
        print("\nInterest Points !!! \n[ Turns , 3_Junc , 4_Junc ] [ ",turns," , ",junc_3," , ",junc_4," ] \n")
        if debug_mapping:
            self.Graph.displaygraph()
            
   # (Graphify) :           Main function 
    #              [Usage : (Convert) Maze ==> Graph]
    def graphify(self,extracted_maze,bot_loc=[],destination=[],car_rect=[]):
        """Performs mapping to convert [(top down) maze(roi) view ==> traversable graph.]

        Args:
            extracted_maze (numpy_1d_array): Occupancy Grid from localization stage [mask]
            bot_loc        (tuple):          Localized robot location.
            destination    (tuple):          User selected location (end).
            car_rect        (list):          Boundingbox of robot,dimensions to be used for reference base size.
            

        Updates:
            self.Graph.graph   => Generated graph from provided maze occupancy grid
             self.maze          => Image displaying only pathways in maze or roi
        """        
        # Check graph extracted or not from the maze
        if not self.graphified:

            # Step 1: Peforming thinning on maze to reduce area to paths that car could follow.
            thinned = thinning(extracted_maze)

            # Step 2: Dilate and Perform thining again to minimize unneccesary interest point (i.e:turns)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
            thinned_dilated = cv2.morphologyEx(thinned, cv2.MORPH_DILATE, kernel)
            _, bw2 = cv2.threshold(thinned_dilated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)        
            thinned = thinning(bw2)
            
            # Step 3: Crop out Boundary that is not part of maze
            thinned_cropped = thinned[self.crp_amt:thinned.shape[0]-self.crp_amt,
                                      self.crp_amt:thinned.shape[1]-self.crp_amt]
            _gps_viz("01_skeleton_road", cv2.cvtColor(thinned_cropped, cv2.COLOR_GRAY2BGR))

            # [NEW]: Estimating start and destination on roadnetwork
            #        from bot_location provided by localization module and destination_loc
            #        provided by user
            if bot_loc!=[]:
                # Use ALL skeleton road pixels for snapping start/end. The old code
                # only used road_cnts[0], which can represent a single contour/branch
                # and can incorrectly snap both start and destination to the same point.
                road_ys, road_xs = np.where(thinned_cropped > 0)
                if road_xs.size == 0:
                    print("[GPS Mapping Debug] ERROR: no road skeleton pixels found")
                    return
                road_pts = np.column_stack((road_xs, road_ys))  # (x=col, y=row)

                bot_loc_in = tuple(bot_loc)
                destination_in = tuple(destination)

                # thinned_cropped is maze_og with crp_amt pixels removed from
                # each side. bot_loc and destination are in the full rotated
                # maze_og frame, so convert them into the cropped frame before
                # nearest-neighbor snapping. Without this, start/end are shifted
                # and can snap to the wrong road branch.
                bot_q_uncropped = (bot_loc_in[0] - self.crp_amt, bot_loc_in[1] - self.crp_amt)
                dst_q_uncropped = (destination_in[0] - self.crp_amt, destination_in[1] - self.crp_amt)

                # Clamp query points into image bounds before nearest-neighbor snap.
                bot_q = (int(np.clip(bot_q_uncropped[0], 0, thinned_cropped.shape[1]-1)),
                         int(np.clip(bot_q_uncropped[1], 0, thinned_cropped.shape[0]-1)))
                dst_q = (int(np.clip(dst_q_uncropped[0], 0, thinned_cropped.shape[1]-1)),
                         int(np.clip(dst_q_uncropped[1], 0, thinned_cropped.shape[0]-1)))

                closest_idx = closest_node(bot_q, road_pts)
                start = (int(road_pts[closest_idx][0]), int(road_pts[closest_idx][1]))

                closest_idx = closest_node(dst_q, road_pts)
                end = (int(road_pts[closest_idx][0]), int(road_pts[closest_idx][1]))

                print("[GPS Mapping Debug] graphify input bot/dst      =", bot_loc_in, destination_in)
                print("[GPS Mapping Debug] graphify clamped bot/dst    =", bot_q, dst_q)
                print("[GPS Mapping Debug] thinned_cropped shape h,w   =", thinned_cropped.shape[0], thinned_cropped.shape[1])
                print("[GPS Mapping Debug] nearest skeleton start/end  =", start, end)

                # If snapped start/end are identical, try to keep the destination as
                # the nearest road pixel among points reasonably far from start.
                if start == end and road_pts.shape[0] > 1:
                    d2_start = np.sum((road_pts - np.array(start))**2, axis=1)
                    far_pts = road_pts[d2_start > 50**2]
                    if far_pts.size > 0:
                        closest_idx = closest_node(dst_q, far_pts)
                        end = (int(far_pts[closest_idx][0]), int(far_pts[closest_idx][1]))
                        print("[GPS Mapping Debug] start==end; re-snapped end  =", end)

                # Visualizing start and end
                thinned_bgr = cv2.cvtColor(thinned_cropped, cv2.COLOR_GRAY2BGR)
                cv2.circle(thinned_bgr, bot_q, 5, (0,0,255), -1)
                cv2.circle(thinned_bgr, dst_q, 5, (255,0,255), -1)
                cv2.circle(thinned_bgr, start, 8, (255,0,0), 2)
                cv2.circle(thinned_bgr, end, 12, (0,255,0), 3)
                cv2.putText(thinned_bgr, "bot_q", (bot_q[0]+8, bot_q[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
                cv2.putText(thinned_bgr, "dst_q", (dst_q[0]+8, dst_q[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1)
                cv2.putText(thinned_bgr, "START snap", (start[0]+8, start[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,0), 1)
                cv2.putText(thinned_bgr, "END snap", (end[0]+8, end[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
                _gps_viz("02_snap_start_end", thinned_bgr)

            # Step 4: Overlay found path on Maze Occupency Grid.
            extracted_maze_cropped = extracted_maze[self.crp_amt:extracted_maze.shape[0]-self.crp_amt,
                                                    self.crp_amt:extracted_maze.shape[1]-self.crp_amt]
            extracted_maze_cropped = cv2.cvtColor(extracted_maze_cropped, cv2.COLOR_GRAY2BGR)
            extracted_maze_cropped[thinned_cropped>0] = (0,255,255)
            
            # Step 5: Identify Interest Points in the path to further reduce processing time
            self.one_pass(thinned_cropped,start,end)

            # Robustness fix: sometimes the user destination is snapped to a road
            # pixel but one_pass() does not register it as Graph.end, leaving
            # Graph.end == 0 and causing path planning to crash with KeyError: 0.
            # Force-add the snapped start/end points and connect them if needed.
            if self.Graph.start == 0:
                start_row, start_col = start[1], start[0]
                print("[GPS Mapping Debug] WARNING: Graph.start was not set by one_pass(); force-adding", (start_row, start_col))
                self.Graph.add_vertex((start_row, start_col), case="_Start_")
                self.Graph.start = (start_row, start_col)
                self.reset_connct_paramtrs()
                self.connect_neighbors(thinned_cropped, start_row, start_col, "_Start_")

            if self.Graph.end == 0:
                end_row, end_col = end[1], end[0]
                print("[GPS Mapping Debug] WARNING: Graph.end was not set by one_pass(); force-adding", (end_row, end_col))
                self.Graph.add_vertex((end_row, end_col), case="_End_")
                self.Graph.end = (end_row, end_col)
                self.reset_connct_paramtrs()
                self.connect_neighbors(thinned_cropped, end_row, end_col, "_End_")

            print("[GPS Mapping Debug] snapped start/end on road =", start, end)
            print("[GPS Mapping Debug] Graph.start/end          =", self.Graph.start, self.Graph.end)
            graph_viz = cv2.cvtColor(thinned_cropped, cv2.COLOR_GRAY2BGR)
            for u, nbrs in self.Graph.graph.items():
                if not isinstance(u, tuple):
                    continue
                up = (int(u[1]), int(u[0]))
                cv2.circle(graph_viz, up, 2, (0, 255, 255), -1)
                for v in nbrs:
                    if v == "case" or not isinstance(v, tuple):
                        continue
                    vp = (int(v[1]), int(v[0]))
                    cv2.line(graph_viz, up, vp, (80, 80, 255), 1)
            if self.Graph.start != 0:
                cv2.circle(graph_viz, (self.Graph.start[1], self.Graph.start[0]), 9, (255,0,0), 2)
            if self.Graph.end != 0:
                cv2.circle(graph_viz, (self.Graph.end[1], self.Graph.end[0]), 12, (0,255,0), 2)
            _gps_viz("03_graph_nodes_edges", graph_viz)
            #cv2.waitKey(0)
            self.maze = thinned_cropped
            self.graphified = True

            # [NEW]: Refining detected dcsn_pts by removing falsePositive/redundant dcsn_pts and displayig them in Colored Image
            [x,y,w,h] = car_rect            
            cnts_dcsn_pts = cv2.findContours(self.maze_dcsn_pts, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]
            # Container to store the refined decision pts [False positive removed]
            refined_dcsn_pts = np.zeros_like(self.maze_dcsn_pts)
            # Rgb image to display refined decision pts
            refined_dcsn_pts_bgr = cv2.cvtColor(thinned_cropped, cv2.COLOR_GRAY2BGR)
            # rgb image to indicate refined decision pts using opencv Shapes
            refined_InterestPoints = cv2.cvtColor(thinned_cropped, cv2.COLOR_GRAY2BGR)

            # loop over each dcsn_pt to weed out False_positives
            for idx,cnt in enumerate(cnts_dcsn_pts):
                (cntr_col,cntr_row) = get_centroid(cnt)
                cntr = (cntr_col,cntr_row)

                # Look at area around the dcsn_pt to check for false positives
                start_row = int(cntr_row-w) if ((cntr_row-w)>=0) else 0
                start_col = int(cntr_col-w) if ((cntr_col-w)>=0) else 0
                end_row = int(cntr_row+w) if ((cntr_row+w)<thinned_cropped.shape[0]) else int(thinned_cropped.shape[0]-1)
                end_col = int(cntr_col+w) if ((cntr_col+w)<thinned_cropped.shape[1]) else int(thinned_cropped.shape[1]-1)

                # Retrieving paths crossing in either of the 4 directions
                lft = thinned_cropped[start_row+1:end_row-1,start_col]
                rgt = thinned_cropped[start_row+1:end_row-1,end_col]
                top = thinned_cropped[start_row        ,start_col+1:end_col-1]
                btm = thinned_cropped[end_row          ,start_col+1:end_col-1]

                # Computing the actual no of paths originating the decision point
                paths = (np.any(lft)*1)+(np.any(rgt)*1)+(np.any(top)*1)+(np.any(btm)*1)

                if paths>2:
                    # Found T-Junc/Intersection
                    cv2.rectangle(refined_dcsn_pts_bgr, (cntr_col-w,cntr_row-w), (cntr_col+w,cntr_row+w), (0,255,0))
                    cv2.drawContours(refined_dcsn_pts, cnts_dcsn_pts, idx, 255)
                    if paths==3:
                        # Indicate T-Junc with triangle
                        refined_InterestPoints = self.triangle(refined_InterestPoints, cntr, 10,(144,140,255),4)
                    else:
                        # Indicate Intersection with rectangle
                        cv2.rectangle(refined_InterestPoints,(cntr_col-10,cntr_row-10) , (cntr_col+10,cntr_row+10), (255,215,0),4)
                else:
                    cv2.rectangle(refined_dcsn_pts_bgr, (cntr_col-w,cntr_row-w), (cntr_col+w,cntr_row+w), (128,0,255))
                # Adding refined decision pts [Intersection and T-Junction] to image of Interest points
                self.maze_interestPts = cv2.bitwise_or(self.maze_interestPts,refined_InterestPoints)
            
            if config.debug and config.debug_mapping:
                cv2.imshow("Extracted_Maze [MazeConverter]",extracted_maze)
                cv2.imshow('Maze (thinned*2)', thinned)
                cv2.imshow('Maze (thinned*2)(Cropped)', thinned_cropped)
                cv2.imshow('Maze (thinned*2)(Cropped)(Path_Overlayed)', extracted_maze_cropped)
                cv2.waitKey(0)
        else:

            if config.debug and config.debug_mapping:
                # [NEW]: Creating Windows for Interest Points and Connection between them
                cv2.namedWindow("Nodes Conected",cv2.WINDOW_FREERATIO)
                cv2.namedWindow("Maze (Interest Points)",cv2.WINDOW_FREERATIO)
                cv2.imshow("Nodes Conected", self.maze_connect)
                cv2.imshow("Maze (Interest Points)", self.maze_interestPts)
            else:
                try:
                    cv2.destroyWindow("Nodes Conected")
                    cv2.destroyWindow("Maze (Interest Points)")
                    cv2.destroyWindow("Extracted_Maze [MazeConverter]")
                    cv2.destroyWindow('Maze (thinned)')
                    cv2.destroyWindow('Maze (thinned*2)')
                    cv2.destroyWindow('Maze (thinned*2)(Cropped)')
                    cv2.destroyWindow('Maze (thinned*2)(Cropped)(Path_Overlayed)')
                except:
                    pass





