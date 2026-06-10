import cv2
import numpy as np

from ...config import config
# ****************************************************  DETECTION ****************************************************
# ****************************************************    LANES   ****************************************************

# Original lane-line pipeline. It expects white mid-lane + yellow outer-lane,
# which worked for the old demo world but is unreliable in the current city road
# where the visible drivable area is mostly black asphalt.
from .a_Segmentation.colour_segmentation_final import Segment_Colour
from .b_Estimation.Our_EstimationAlgo import Estimate_MidLane
from .c_Cleaning.CheckifYellowLaneCorrect_RetInnerBoundary import GetYellowInnerEdge
from .c_Cleaning.ExtendLanesAndRefineMidLaneEdge import ExtendShortLane
from .d_LaneInfo_Extraction.GetStateInfoandDisplayLane import FetchInfoAndDisplay
from .utilities import findlaneCurvature

road_debug_counter = 0


def _largest_contour_mask(mask, min_area=400):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        if not cnts:
                return None
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) < min_area:
                return None
        out = np.zeros_like(mask)
        cv2.drawContours(out, [cnt], -1, 255, -1)
        return out


def _front_road_roi(shape):
        """Trapezoid ROI for the drivable road directly in front of the car.

        This prevents dark curb/grass/shadow at the image sides from being
        treated as road when the car is close to the curb.
        """
        h, w = shape[:2]
        roi = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[
                (int(w * 0.12), h - 1),
                (int(w * 0.88), h - 1),
                (int(w * 0.68), int(h * 0.18)),
                (int(w * 0.32), int(h * 0.18)),
        ]], dtype=np.int32)
        cv2.fillPoly(roi, pts, 255)
        return roi


def _component_near_bottom_center(mask):
        """Keep only the dark-road component connected near car nose.

        The largest dark component may be a curb/shadow/sidewalk when the car is
        near the road edge. We instead look for a connected component overlapping
        a small seed area around the bottom center of the image.
        """
        h, w = mask.shape[:2]
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
                return None

        seed = np.zeros_like(mask)
        cv2.rectangle(seed,
                      (int(w * 0.40), int(h * 0.70)),
                      (int(w * 0.60), h - 1),
                      255, -1)
        seed_labels = labels[seed > 0]
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size == 0:
                return None

        # Choose the seed-overlapping component with largest area.
        candidate_labels = np.unique(seed_labels)
        best_label = max(candidate_labels, key=lambda lab: stats[lab, cv2.CC_STAT_AREA])
        area = stats[best_label, cv2.CC_STAT_AREA]
        if area < int(h * w * 0.03):
                return None

        out = np.zeros_like(mask)
        out[labels == best_label] = 255
        return out


def _row_center(mask, row, prefer_x=None, prefer_widest=False):
        """Return center x of the road segment most likely under the car.

        Earlier code chose the widest continuous dark segment. In the city world,
        parked cars, intersections, or road branches can make a side segment wider
        than the lane directly in front of the car, causing a persistent right/left
        steering bias. Prefer the segment that contains the image center; otherwise
        choose the segment whose center is closest to the image center.
        """
        row = max(0, min(mask.shape[0] - 1, int(row)))
        xs = np.where(mask[row] > 0)[0]
        if xs.size == 0:
                return None

        if prefer_x is None:
                prefer_x = int(mask.shape[1] / 2)

        breaks = np.where(np.diff(xs) > 1)[0]
        starts = np.r_[0, breaks + 1]
        ends = np.r_[breaks, xs.size - 1]

        segments = []
        for st, en in zip(starts, ends):
                x0, x1 = int(xs[st]), int(xs[en])
                width = x1 - x0 + 1
                center = int((x0 + x1) / 2)
                contains_center = (x0 <= prefer_x <= x1)
                segments.append((contains_center, abs(center - prefer_x), -width, x0, x1, center, width))

        if prefer_widest:
                # For road-region following, choose the widest black asphalt
                # segment. If the car is already on the shoulder/curb, choosing
                # the segment closest to image center locks onto the shoulder
                # and makes the vehicle spin farther away from the lane.
                best = sorted(segments, key=lambda v: (v[2], v[1]))[0]
        else:
                # Sort priority:
                # 1) segment containing image center
                # 2) nearest segment to image center
                # 3) wider segment
                best = sorted(segments, key=lambda v: (not v[0], v[1], v[2]))[0]
        return best[5], best[6]


def _detect_black_road(img):
        global road_debug_counter
        """Follow the black asphalt region instead of white/yellow lane lines.

        Returns distance, curvature, debug_frame. distance is road center near
        car nose minus image center; curvature approximates road direction.
        """
        img_cropped = img[config.CropHeight_resized:, :]
        hls = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)

        # Black asphalt is both dark and low-saturation. The older mask used
        # only low lightness, so it also accepted gray curbs, shadows and dark
        # green/yellow shoulders. Intersect HLS and HSV masks to keep only the
        # nearly-black road surface.
        max_l = int(getattr(config, "road_black_max_hls_l", 62))
        max_hls_s = int(getattr(config, "road_black_max_hls_s", 110))
        max_v = int(getattr(config, "road_black_max_hsv_v", 85))
        max_hsv_s = int(getattr(config, "road_black_max_hsv_s", 95))
        mask_hls = cv2.inRange(hls, np.array([0, 0, 0]), np.array([255, max_l, max_hls_s]))
        mask_hsv = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([255, max_hsv_s, max_v]))
        mask = cv2.bitwise_and(mask_hls, mask_hsv)

        # Remove small objects and connect road patches.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # Prefer road connected to the bottom-center near the car nose. Do not
        # clip the road by the trapezoid before measuring its center: the old
        # code measured the ROI center, so distance stayed 0 even when the car
        # drifted toward the curb.
        front_roi = _front_road_roi(mask.shape)
        road = _component_near_bottom_center(mask)
        if road is None:
                road = _largest_contour_mask(cv2.bitwise_and(mask, front_roi), min_area=int(mask.shape[0] * mask.shape[1] * 0.03))
        if road is None:
                return -1000, -1000, img_cropped

        h, w = road.shape[:2]
        img_center = int(w / 2)

        # Use several rows and median them to avoid a single obstacle/branch making
        # the car overreact. Prefer road segments around image center.
        low_samples = []
        for frac in (0.58, 0.66, 0.74):
                c = _row_center(road, int(h * frac), prefer_x=img_center, prefer_widest=True)
                if c is not None:
                        low_samples.append((c[0], c[1], int(h * frac)))
        if not low_samples:
                return -1000, -1000, img_cropped

        low_x = int(np.median([v[0] for v in low_samples]))
        low_width = int(np.median([v[1] for v in low_samples]))
        low_y = int(np.median([v[2] for v in low_samples]))

        up = _row_center(road, int(h * 0.38), prefer_x=low_x, prefer_widest=True)
        if up is None:
                up_x, up_y = low_x, low_y
        else:
                up_x, _ = up
                up_y = int(h * 0.38)

        # Ignore rows where the visible road width is tiny; likely an obstacle or bad segmentation.
        min_width = int(w * float(getattr(config, "road_min_width_ratio", 0.35)))
        if low_width < min_width:
                return -1000, -1000, img_cropped

        distance = low_x - img_center
        # Deadband: don't steer for tiny center offset.
        if abs(distance) < 8:
                distance = 0
        # Clamp distance so a bad segmentation frame cannot command a huge turn.
        distance = int(np.clip(distance, -45, 45))

        curvature = findlaneCurvature(low_x, low_y, up_x, up_y)

        # If the detected asphalt spans almost the whole image width near the car,
        # there is no reliable left/right road edge in the camera view. In this
        # case the far/top point is often selected from an intersection/branch or
        # a partial dark region, producing a fake huge curvature such as -67 deg.
        # Treat this as "road fills FOV -> keep straight" unless the car is
        # clearly off-center.
        if low_width > int(w * 0.90) and abs(distance) <= 8:
                up_x = low_x
                up_y = max(0, int(h * 0.38))
                curvature = 0.0

        # Final safety clamp: lane follower should not command extreme steering
        # from vision-only curvature in this city world. GPS/SatNav handles large
        # route turns separately.
        curvature = float(np.clip(curvature, -25.0, 25.0))

        # Debug overlay on the cropped frame used by Bot View.
        if config.Testing:
                overlay = img_cropped.copy()
                road_color = np.zeros_like(overlay)
                road_color[:, :, 1] = road
                overlay = cv2.addWeighted(overlay, 1.0, road_color, 0.25, 0)
                # Draw forward ROI boundary in cyan so we can verify side curbs are excluded.
                roi_cnts = cv2.findContours(front_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
                cv2.drawContours(overlay, roi_cnts, -1, (255, 255, 0), 1)
                cv2.circle(overlay, (low_x, low_y), 5, (0, 255, 255), -1)
                cv2.circle(overlay, (up_x, up_y), 5, (255, 0, 0), -1)
                cv2.line(overlay, (low_x, low_y), (up_x, up_y), (255, 0, 0), 2)
                cv2.line(overlay, (int(w/2), h), (int(w/2), int(h*0.65)), (0, 0, 255), 2)
                cv2.putText(overlay, "dist={} curv={:.1f}".format(distance, curvature),
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                cv2.putText(overlay, "low=({}, {}) up=({}, {}) width={}".format(low_x, low_y, up_x, up_y, low_width),
                            (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                # Dedicated debug windows. Green/white means detected asphalt road.
                cv2.imshow("[RoadFollow] road_mask", road)
                cv2.imshow("[RoadFollow] overlay", overlay)

                # Put overlay back into img so the existing Bot View shows the new road following.
                img[config.CropHeight_resized:, :] = overlay

                road_debug_counter += 1
                if road_debug_counter % 15 == 0:
                        print("[RoadFollow Debug] distance={} curvature={:.2f} low=({}, {}) up=({}, {}) road_width={}".format(
                                distance, curvature, low_x, low_y, up_x, up_y, low_width))

        return distance, curvature, img_cropped


def _detect_original_lane_lines(img):
        img_cropped = img[config.CropHeight_resized:, :]
        Mid_edge_ROI,Mid_ROI_mask,Outer_edge_ROI,OuterLane_TwoSide,OuterLane_Points = Segment_Colour(img_cropped,config.minArea_resized)
        Estimated_midlane = Estimate_MidLane(Mid_edge_ROI,config.MaxDist_resized)
        OuterLane_OneSide,Outer_cnts_oneSide,Mid_cnts,Offset_correction = GetYellowInnerEdge(OuterLane_TwoSide,Estimated_midlane,OuterLane_Points)
        Estimated_midlane,OuterLane_OneSide = ExtendShortLane(Estimated_midlane,Mid_cnts,Outer_cnts_oneSide,OuterLane_OneSide)
        Distance , Curvature = FetchInfoAndDisplay(Mid_edge_ROI,Estimated_midlane,OuterLane_OneSide,img_cropped,Offset_correction)
        return Distance, Curvature


def detect_Lane(img):
        """Extract steering information.

        In the current city world, roads are black asphalt and lane lines are not
        reliable/consistent. Use road-region following by default. If it fails,
        fall back to the old white/yellow lane-line detector.
        """
        Distance, Curvature, _ = _detect_black_road(img)
        if Distance != -1000 and Curvature != -1000:
                return Distance, Curvature

        # Fallback for old worlds with white/yellow lane markings.
        return _detect_original_lane_lines(img)
