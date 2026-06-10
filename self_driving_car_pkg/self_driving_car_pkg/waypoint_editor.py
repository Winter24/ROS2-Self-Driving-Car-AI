import cv2
import math
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .GPS_Navigation.manual_map import save_manual_map, load_manual_map, default_map_path


class WaypointEditor(Node):
    def __init__(self):
        super().__init__('gps_waypoint_editor')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/upper_camera/image_raw', self.on_image, 10)
        self.frame = None
        # Editor does not need vision localization. The satellite camera is a
        # 90-degree rotated top-down view, so road-network coordinates used by
        # the current GPS code are approximately:
        #   road_x = sat_y
        #   road_y = image_width - sat_x
        # This avoids crashing inside bot_localization while editing waypoints.
        self.have_transform = True
        self.map_path = default_map_path()
        try:
            self.data = load_manual_map(self.map_path)
            self.get_logger().info(f"Loaded existing map: {self.map_path}")
        except Exception:
            self.data = {"version": 1, "nodes": [], "edges": []}
        self.selected = []
        self.next_id = 1 + max([n.get('id', 0) for n in self.data.get('nodes', [])] or [0])
        cv2.namedWindow('GPS Waypoint Editor', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('GPS Waypoint Editor', self.on_mouse)
        self.get_logger().info('Controls: LEFT add/select node | C connect selected/last two | D delete selected | S save | R reset | Q quit')

    def on_image(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.draw()

    def road_xy_from_sat(self, sat_xy):
        if self.frame is None:
            return None
        x, y = sat_xy
        h, w = self.frame.shape[:2]
        return (int(y), int(w - x))

    def nearest_node(self, x, y, radius=14):
        best = None
        best_d = radius
        for n in self.data['nodes']:
            nx, ny = n['sat_xy']
            d = math.hypot(nx - x, ny - y)
            if d < best_d:
                best = n
                best_d = d
        return best

    def add_node(self, x, y):
        road_xy = self.road_xy_from_sat((x, y))
        n = {"id": self.next_id, "sat_xy": [int(x), int(y)], "road_xy": [int(road_xy[0]), int(road_xy[1])] if road_xy else None}
        self.next_id += 1
        self.data['nodes'].append(n)
        print(f"[WaypointEditor] add node id={n['id']} sat_xy={n['sat_xy']} road_xy={n['road_xy']}")
        return n

    def edge_exists(self, a, b):
        return any((e['from'] == a and e['to'] == b) or (not e.get('one_way', False) and e['from'] == b and e['to'] == a) for e in self.data['edges'])

    def connect(self, a, b):
        if a == b or self.edge_exists(a, b):
            return
        nodes = {n['id']: n for n in self.data['nodes']}
        ax, ay = nodes[a]['road_xy'] or nodes[a]['sat_xy']
        bx, by = nodes[b]['road_xy'] or nodes[b]['sat_xy']
        cost = math.hypot(ax - bx, ay - by)
        self.data['edges'].append({"from": int(a), "to": int(b), "cost": float(cost), "one_way": False})
        print(f"[WaypointEditor] connect {a} <-> {b} cost={cost:.1f}")

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        n = self.nearest_node(x, y)
        if n is None:
            n = self.add_node(x, y)
        if n['id'] in self.selected:
            self.selected.remove(n['id'])
        else:
            self.selected.append(n['id'])
            self.selected = self.selected[-2:]
        print(f"[WaypointEditor] selected={self.selected}")
        self.draw()

    def draw(self):
        if self.frame is None:
            return
        img = self.frame.copy()
        nodes = {n['id']: n for n in self.data['nodes']}
        for e in self.data['edges']:
            if e['from'] in nodes and e['to'] in nodes:
                p1 = tuple(nodes[e['from']]['sat_xy'])
                p2 = tuple(nodes[e['to']]['sat_xy'])
                cv2.line(img, p1, p2, (0, 0, 255), 3)
                cv2.line(img, p1, p2, (255, 255, 255), 1)
        for n in self.data['nodes']:
            p = tuple(n['sat_xy'])
            color = (0, 255, 255) if n['id'] not in self.selected else (0, 255, 0)
            cv2.circle(img, p, 8, color, -1)
            cv2.circle(img, p, 10, (0, 0, 0), 1)
            cv2.putText(img, str(n['id']), (p[0] + 10, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(img, 'LEFT add/select | C connect | D delete | S save | R reset | Q quit', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.imshow('GPS Waypoint Editor', img)
        k = cv2.waitKey(1) & 0xff
        if k in (ord('q'), 27):
            rclpy.shutdown()
        elif k == ord('s'):
            path = save_manual_map(self.data, self.map_path)
            print(f"[WaypointEditor] saved {path}")
        elif k == ord('c'):
            if len(self.selected) == 2:
                self.connect(self.selected[0], self.selected[1])
            elif len(self.data['nodes']) >= 2:
                self.connect(self.data['nodes'][-2]['id'], self.data['nodes'][-1]['id'])
        elif k == ord('d'):
            ids = set(self.selected)
            self.data['nodes'] = [n for n in self.data['nodes'] if n['id'] not in ids]
            self.data['edges'] = [e for e in self.data['edges'] if e['from'] not in ids and e['to'] not in ids]
            self.selected = []
        elif k == ord('r'):
            self.data = {"version": 1, "nodes": [], "edges": []}
            self.selected = []
            self.next_id = 1


def main(args=None):
    rclpy.init(args=args)
    node = WaypointEditor()
    try:
        rclpy.spin(node)
    finally:
        try:
            save_manual_map(node.data, node.map_path)
            print(f"[WaypointEditor] auto-saved {node.map_path}")
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
