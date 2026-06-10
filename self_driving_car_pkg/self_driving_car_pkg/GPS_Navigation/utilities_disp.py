import cv2
import numpy as np
from math import pi,cos,sin
from . import config


# Overlay detected regions over the bot_view
def overlay(image,overlay_img):

    gray = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)


    roi = image
    img2 = overlay_img
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    
    image = img1_bg + img2_fg
    return image

# Overlay detected regions (User-specified-amount) over the frame_disp
def overlay_cropped(frame_disp,image_rot,crop_loc_row,crop_loc_col,overlay_cols):
    
    image_rot_cols = image_rot.shape[1]
    gray = cv2.cvtColor(image_rot[:,image_rot_cols-overlay_cols:image_rot_cols], cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)

    frame_overlay_cols = crop_loc_col + image_rot_cols
    roi = frame_disp[crop_loc_row:crop_loc_row + image_rot.shape[0],frame_overlay_cols-overlay_cols:frame_overlay_cols]            
    img2 = image_rot[:,image_rot_cols-overlay_cols:image_rot_cols]

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    
    frame_disp[crop_loc_row:crop_loc_row + image_rot.shape[0],frame_overlay_cols-overlay_cols:frame_overlay_cols] = img1_bg + img2_fg


def overlay_live(frame_disp,overlay,overlay_map,overlay_path,transform_arr,crp_amt):

    overlay_rot = cv2.rotate(overlay, cv2.ROTATE_90_CLOCKWISE)
    map_rot = cv2.rotate(overlay_map, cv2.ROTATE_90_CLOCKWISE)
    image_rot = cv2.rotate(overlay_path, cv2.ROTATE_90_CLOCKWISE)

    crop_loc_col = transform_arr[0]+crp_amt
    #crop_loc_endCol = transform_arr[0]+transform_arr[2]+crp_amt
    crop_loc_row = transform_arr[1]+crp_amt

    new_cols = int(overlay_rot.shape[1]*config.debug_live_amount)
    new_path_cols = int(overlay_rot.shape[1]*config.debug_path_live_amount)
    new_map_cols = int(overlay_rot.shape[1]*config.debug_map_live_amount)


    frame_disp[crop_loc_row:crop_loc_row + overlay_rot.shape[0],crop_loc_col:crop_loc_col + new_cols] = overlay_rot[:,0:new_cols]
    
    if config.debug_map_live_amount>0:
        overlay_cropped(frame_disp,map_rot,crop_loc_row,crop_loc_col,new_map_cols)
    if config.debug_path_live_amount>0:
        overlay_cropped(frame_disp,image_rot,crop_loc_row,crop_loc_col,new_path_cols)


def overlay_route_live(frame_disp, overlay_path, transform_arr, crp_amt):
    """Always draw the selected GPS route on SatView live.

    The old route overlay was hidden unless Debug_Live and Debug_path(Live)
    trackbars were enabled. This helper draws choosen_route directly on the
    satellite frame using the same transform as overlay_live().
    """
    if overlay_path is None or not hasattr(overlay_path, 'shape') or len(overlay_path) == 0:
        return

    image_rot = cv2.rotate(overlay_path, cv2.ROTATE_90_CLOCKWISE)
    crop_loc_col = int(transform_arr[0] + crp_amt)
    crop_loc_row = int(transform_arr[1] + crp_amt)

    # Clip overlay to frame bounds safely.
    y0 = max(0, crop_loc_row)
    x0 = max(0, crop_loc_col)
    y1 = min(frame_disp.shape[0], crop_loc_row + image_rot.shape[0])
    x1 = min(frame_disp.shape[1], crop_loc_col + image_rot.shape[1])
    if y1 <= y0 or x1 <= x0:
        return

    oy0 = y0 - crop_loc_row
    ox0 = x0 - crop_loc_col
    route_crop = image_rot[oy0:oy0 + (y1-y0), ox0:ox0 + (x1-x0)]

    gray = cv2.cvtColor(route_crop, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]
    if cv2.countNonZero(mask) == 0:
        return

    # Thicken route for visibility.
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    route_visible = np.zeros_like(route_crop)
    route_visible[:, :] = (0, 0, 255)  # red route in BGR
    route_visible = cv2.bitwise_and(route_visible, route_visible, mask=mask)

    roi = frame_disp[y0:y1, x0:x1]
    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.addWeighted(roi, 0.25, route_visible, 0.95, 0)
    fg = cv2.bitwise_and(fg, fg, mask=mask)
    frame_disp[y0:y1, x0:x1] = cv2.add(bg, fg)

# Draw speedometer and arrows indicating bot speed and direction at given moment
def draw_bot_speedo(image,bot_speed,bot_turning):
    height, width = image.shape[0:2]
    # Ellipse parameters
    radius = 50
    center = (int(width / 2), height - 25)
    axes = (radius, radius)
    angle = 0
    startAngle = 180
    endAngle = 360
    thickness = 10

    # http://docs.opencv.org/modules/core/doc/drawing_functions.html#ellipse
    cv2.ellipse(image, center, axes, angle, startAngle, endAngle, (0,0,0), thickness)

    Estimted_line = np.zeros_like(image)
    max_speed = 1.5
    angle = -(((bot_speed/max_speed)*180)+90)
    speed_mph = int((bot_speed/max_speed)*200)
    length = 300
    P1 = center
    
    P2 = ( 
            int(P1[0] + length * sin(angle * (pi / 180.0) ) ),
            int(P1[1] + length * cos(angle * (pi / 180.0) ) ) 
            )
    
    cv2.line(Estimted_line,center, P2, (255,255,255),3)
    meter_mask = np.zeros((image.shape[0],image.shape[1]),np.uint8)

    cv2.ellipse(meter_mask, center, axes, angle, 0, endAngle, 255, -1)

    Estimted_line = cv2.bitwise_and(Estimted_line, Estimted_line,mask = meter_mask)

    speed_clr = (0,0,0)
    if speed_mph<20:
        speed_clr = (0,255,255)
    elif speed_mph<40:
        speed_clr = (0,255,0)
    elif speed_mph<60:
        speed_clr = (0,140,255)
    elif speed_mph>=60:
        speed_clr = (0,0,255)
    cv2.putText(image, str(speed_mph), (center[0]+10,center[1]-10), cv2.FONT_HERSHEY_PLAIN, 2, speed_clr,3)


    image = overlay(image,Estimted_line)
    if bot_turning>0.2:
        image = cv2.arrowedLine(image, (40,int(image.shape[0]/2)), (10,int(image.shape[0]/2)),
                                        (0,140,255), 13,tipLength=0.8)
    elif bot_turning<-0.2:
        image = cv2.arrowedLine(image, (image.shape[1]-40,int(image.shape[0]/2)), (image.shape[1]-10,int(image.shape[0]/2)),
                                        (0,140,255), 13,tipLength=0.8)

    return image


def disp_SatNav(frame_disp,sbot_view,bot_curr_speed,bot_curr_turning,maze_interestPts,choosen_route,img_choosen_route,transform_arr,crp_amt):
    """Draw a cleaner live GPS UI directly on the satellite map.

    Old UI put the map back inside a tablet image and used a very large bot view,
    which made the navigation display hard to read. This version keeps the map as
    the main canvas and adds a compact bot-camera picture-in-picture.
    """
    # Always draw the planned GPS route on the live satellite view.
    overlay_route_live(frame_disp, choosen_route, transform_arr, crp_amt)

    # Optional old debug overlays remain available via trackbars.
    if config.debug_live:
        overlay_live(frame_disp,img_choosen_route,maze_interestPts,choosen_route,transform_arr,crp_amt)

    # Compact bot view; keep it small so it does not cover the road network.
    bot_view = cv2.resize(sbot_view, None, fx=0.45, fy=0.45)
    bot_view = draw_bot_speedo(bot_view, bot_curr_speed, bot_curr_turning)

    margin = 12
    label_h = 26
    border = 3
    bot_h, bot_w = bot_view.shape[:2]

    x0 = margin
    y0 = frame_disp.shape[0] - bot_h - label_h - margin
    if y0 < margin:
        y0 = margin
    x1 = min(x0 + bot_w, frame_disp.shape[1] - margin)
    y1 = min(y0 + label_h + bot_h, frame_disp.shape[0] - margin)

    # If the frame is too small, crop bot view to fit safely.
    fit_w = x1 - x0
    fit_h = y1 - y0 - label_h
    bot_view = bot_view[:fit_h, :fit_w]
    bot_h, bot_w = bot_view.shape[:2]
    x1 = x0 + bot_w
    y1 = y0 + label_h + bot_h

    # Semi-transparent dark panel for readability.
    overlay_panel = frame_disp.copy()
    cv2.rectangle(overlay_panel, (x0-border, y0-border), (x1+border, y1+border), (0,0,0), -1)
    cv2.addWeighted(overlay_panel, 0.55, frame_disp, 0.45, 0, frame_disp)

    # Header and border.
    cv2.rectangle(frame_disp, (x0-border, y0-border), (x1+border, y0+label_h), (35,35,35), -1)
    cv2.putText(frame_disp, "Bot View", (x0+8, y0+19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.rectangle(frame_disp, (x0-border, y0-border), (x1+border, y1+border), (0,255,255), border)
    frame_disp[y0+label_h:y0+label_h+bot_h, x0:x0+bot_w] = bot_view

    # Small status text at top-left.
    status = "GPS Nav | speed {:.2f} | turn {:.2f}".format(float(bot_curr_speed), float(bot_curr_turning))
    cv2.rectangle(frame_disp, (8, 8), (430, 38), (0,0,0), -1)
    cv2.putText(frame_disp, status, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
