import json
import math
import os
import heapq


def default_map_path():
    return os.path.join(os.path.dirname(__file__), "resource", "manual_gps_map.json")


def load_manual_map(path=None):
    path = path or default_map_path()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = json.load(f)
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    return data


def save_manual_map(data, path=None):
    path = path or default_map_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def node_xy(node, key="road_xy"):
    xy = node.get(key) or node.get("sat_xy")
    return (float(xy[0]), float(xy[1]))


def dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def nearest_node_id(data, point_xy, key="road_xy"):
    best = None
    best_d = 1e18
    for n in data.get("nodes", []):
        xy = node_xy(n, key)
        d = dist(xy, point_xy)
        if d < best_d:
            best_d = d
            best = n["id"]
    return best, best_d


def shortest_node_path(data, start_id, end_id, key="road_xy"):
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    adj = {nid: [] for nid in nodes}
    for e in data.get("edges", []):
        a, b = e["from"], e["to"]
        if a not in nodes or b not in nodes:
            continue
        w = e.get("cost")
        if w is None:
            w = dist(node_xy(nodes[a], key), node_xy(nodes[b], key))
        adj[a].append((b, float(w)))
        if not e.get("one_way", False):
            adj[b].append((a, float(w)))

    pq = [(0.0, start_id)]
    prev = {start_id: None}
    cost = {start_id: 0.0}
    while pq:
        c, u = heapq.heappop(pq)
        if u == end_id:
            break
        if c != cost.get(u):
            continue
        for v, w in adj.get(u, []):
            nc = c + w
            if nc < cost.get(v, 1e18):
                cost[v] = nc
                prev[v] = u
                heapq.heappush(pq, (nc, v))
    if end_id not in prev:
        return []
    path = []
    cur = end_id
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def path_road_xy(data, node_ids):
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    return [tuple(nodes[i]["road_xy"]) for i in node_ids if i in nodes and "road_xy" in nodes[i]]


def path_sat_xy(data, node_ids):
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    return [tuple(nodes[i]["sat_xy"]) for i in node_ids if i in nodes and "sat_xy" in nodes[i]]
