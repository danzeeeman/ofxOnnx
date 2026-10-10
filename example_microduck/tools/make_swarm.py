"""Make the shared world of the swarm: N microducks in one MuJoCo model, inside a walled arena.

    python make_swarm.py <data_dir> <navigator.onnx> [ducks=16] [half_side=2.0] [obstacles=20]

Needs `pip install onnx` (for the policy) and reads <data_dir>/microduck/microduck.xml, which make_microduck.py wrote.
Writes next to it:
  swarm.xml        N copies of the microduck (every body, joint, site, actuator and sensor name prefixed d0_, d1_, ...), a
                   floor and four walls (0.1 m thick, 0.3 m high: higher than the time-of-flight sensor on the chest, so
                   that the sensor sees them), as Isaac Lab's enclosed navigation task has them
  navigator.onnx   the policy that drives every duck (Isaac-Skill-NavigateWalkEnclosed-Microduck), weights embedded

The ducks collide with each other, the floor and the walls, but not with themselves (the training had self collision off).
That takes one contact bit per duck: duck k has contype = floor bit | its own bit, and conaffinity = every other duck's
bit, so two ducks collide when one's type meets the other's affinity, and one duck's parts never do.
"""
import copy, os, sys
import xml.etree.ElementTree as ET

data_dir, policy = sys.argv[1:3]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
half = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
num_obstacles = int(sys.argv[5]) if len(sys.argv) > 5 else 20
folder = os.path.join(data_dir, "microduck")
assert n <= 30, "one contact bit per duck"

tree = ET.parse(os.path.join(folder, "microduck.xml"))
root = tree.getroot()
world = root.find("worldbody")
trunk = [b for b in world if b.tag == "body"][0]
actuators, sensors = root.find("actuator"), root.find("sensor")
floor = [g for g in world if g.tag == "geom" and g.get("name") == "floor"][0]
assert floor.get("conaffinity", "1") == "1" and floor.get("contype") == "0", floor.attrib   # the floor is what ducks' bit 0 meets

# the ducks
world.remove(trunk)
old_act, old_sen = list(actuators), list(sensors)
for e in old_act: actuators.remove(e)
for e in old_sen: sensors.remove(e)
all_bits = 0
for k in range(n): all_bits |= 1 << (k + 1)

def prefixed(e, p):
    for el in e.iter():
        if el.get("name") is not None: el.set("name", p + el.get("name"))

for k in range(n):
    p = "d%d_" % k
    body = copy.deepcopy(trunk)
    prefixed(body, p)
    mine = 1 << (k + 1)
    for g in body.iter("geom"):
        if g.get("contype") != "0" or g.get("conaffinity") != "0":   # a collision shape
            g.set("contype", str(1 | mine))                          # meets the floor and the walls (bit 0), and the others' affinity
            g.set("conaffinity", str(all_bits & ~mine))              # meets the other ducks, never its own parts
    world.append(body)
    for a in old_act:
        a = copy.deepcopy(a); a.set("name", p + a.get("name")); a.set("joint", p + a.get("joint")); actuators.append(a)
    for s in old_sen:
        s = copy.deepcopy(s); s.set("name", p + s.get("name")); s.set("site", p + s.get("site")); sensors.append(s)

# the walls (Isaac Lab's _wall_cfg): a square enclosure with the half side `half`
thick, height = 0.1, 0.3
length = 2 * half + 2 * thick
for i in range(4):
    sign = 1.0 if i % 2 == 0 else -1.0
    off = sign * (half + thick / 2)
    if i < 2: size, pos = (length / 2, thick / 2, height / 2), (0.0, off, height / 2)
    else:     size, pos = (thick / 2, length / 2, height / 2), (off, 0.0, height / 2)
    ET.SubElement(world, "geom", name="wall%d" % i, type="box", size="%g %g %g" % size, pos="%g %g %g" % pos,
                  rgba="0.6 0.6 0.65 1", contype="0", conaffinity="1", friction="1 0.005 0.0001")

# the obstacles: the three kinds of the enclosed navigation task (a box, a post, a long low wall) and five more, in turn. All are
# higher than the time-of-flight sensor (0.14 m) so that the ducks can see them. Placed at random but always the same (seeded),
# spread apart, and clear of the middle. Each kind is a list of parts: (type, size, offset in the obstacle's own frame, colour).
import math, random
random.seed(7)
kinds = [
    ("box",      [("box", (0.10, 0.10, 0.125), (0, 0), "0.8 0.3 0.3 1")], 0.15),                                # task: a box
    ("post",     [("cylinder", (0.07, 0.15), (0, 0), "0.3 0.5 0.8 1")], 0.07),                                  # task: a post
    ("wall",     [("box", (0.25, 0.04, 0.125), (0, 0), "0.4 0.7 0.4 1")], 0.25),                                # task: a low wall
    ("column",   [("cylinder", (0.15, 0.14), (0, 0), "0.9 0.6 0.2 1")], 0.15),                                  # a fat round column
    ("pillar",   [("box", (0.06, 0.06, 0.175), (0, 0), "0.6 0.4 0.8 1")], 0.09),                                # a tall slim pillar
    ("barrier",  [("box", (0.45, 0.03, 0.125), (0, 0), "0.5 0.6 0.75 1")], 0.45),                               # a long thin wall
    ("corner",   [("box", (0.20, 0.035, 0.125), (0, -0.115), "0.9 0.85 0.3 1"), ("box", (0.035, 0.15, 0.125), (-0.165, 0.035), "0.9 0.85 0.3 1")], 0.25),   # an L
    ("stick",    [("cylinder", (0.035, 0.15), (0, 0), "0.9 0.5 0.6 1")], 0.035),                                # a thin post
]
placed = []
for i in range(num_obstacles):
    name, parts, reach = kinds[i % len(kinds)]
    for gap in (0.4, 0.3, 0.25, 0.2):                       # the clear gap between two obstacles: as wide as the crowd of them allows
        found = False
        for _ in range(1500):
            x, y = random.uniform(-half + 0.45, half - 0.45), random.uniform(-half + 0.45, half - 0.45)
            if math.hypot(x, y) < 0.4: continue
            if all(math.hypot(x - px, y - py) >= gap + reach + pr for px, py, pr in placed): found = True; break
        if found: break
    placed.append((x, y, reach))
    yaw = random.uniform(0, math.pi)
    for j, (kind, size, (ox, oy), rgba) in enumerate(parts):
        hz = size[2] if kind == "box" else size[1]
        wx, wy = x + ox * math.cos(yaw) - oy * math.sin(yaw), y + ox * math.sin(yaw) + oy * math.cos(yaw)
        ET.SubElement(world, "geom", name="obstacle%d_%s%d" % (i, name, j), type=kind, size=" ".join("%g" % v for v in size),
                      pos="%g %g %g" % (wx, wy, hz), euler="0 0 %g" % yaw, rgba=rgba, contype="0", conaffinity="1",
                      friction="1 0.005 0.0001")
print("%d obstacles (%d kinds)" % (num_obstacles, len(kinds)))

ET.indent(tree, space="  ")
tree.write(os.path.join(folder, "swarm.xml"), encoding="utf-8")
print("swarm.xml: %d ducks, a %g m x %g m arena" % (n, 2 * half, 2 * half))

import onnx
m = onnx.load(policy)
onnx.save(m, os.path.join(folder, "navigator.onnx"), save_as_external_data=False)
print("navigator.onnx: %d bytes, weights embedded" % os.path.getsize(os.path.join(folder, "navigator.onnx")))
