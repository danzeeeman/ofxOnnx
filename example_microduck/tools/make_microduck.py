"""Make the data the microduck example loads, from the robot's URDF and its trained walking policy.

    python make_microduck.py <microduck.urdf> <assets_parent_dir> <walker.onnx> <out_data_dir>

Needs `pip install mujoco==3.13.0 onnx` (the version of the MuJoCo library in ofxRobot/libs/mujoco: a model written by a
newer MuJoCo can use attributes an older one rejects).

Writes into <out_data_dir>/microduck/:
  microduck.xml      the MuJoCo model: the URDF with a free joint on the trunk, the physics the policy was trained
                     with (position actuators kp 2.0 / kd 0.1, armature 0.005, friction 1.0, robot geoms do not
                     collide with each other), and a floor
  assets/*.stl       the meshes it refers to
  left_leg.joints.json, right_leg.joints.json, head.joints.json
                     ofxRobot joint descriptors (RobotKinematics): one serial chain each, from the trunk
  microduck.json     what the code needs to know: joint order, default pose, action scales, gains, timing
  walker.onnx        the policy, weights embedded (one file)

The order of joints matters and is not written down anywhere in the training setup: the policy's 14 inputs and 14
outputs follow Isaac Lab's articulation order, which for this robot turns out to be the URDF's own depth-first
order (left leg, head, right leg). That is what the policy was found to walk with: any other order falls over.
"""
import json, math, os, re, shutil, sys
import xml.etree.ElementTree as ET
import mujoco

urdf_path, assets_parent, policy, out_data = sys.argv[1:5]
out = os.path.join(out_data, "microduck")
os.makedirs(os.path.join(out, "assets"), exist_ok=True)

# ---- the physics the policy was trained with (IsaacLab env.yaml, Isaac-Velocity-Flat-Microduck)
KP, KD, ARMATURE = 2.0, 0.1, 0.005
SIM_DT, DECIMATION = 0.005, 4
LEG_SCALE, HEAD_SCALE = 0.25, 0.5
DEFAULT_POS = {"left_hip_roll": -0.0873, "right_hip_roll": 0.0873, "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
               "left_knee": -0.0049, "right_knee": 0.0049, "left_ankle": 0.453, "right_ankle": -0.453,
               "neck_pitch": 0.3491, "head_pitch": 0.3491}   # every other joint 0
INIT_HEIGHT = 0.117
COMMAND_RANGE = {"vx": [-0.2, 0.4], "vy": [-0.2, 0.2], "wz": [-1.0, 1.0]}
LEG_JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"]
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]

root = ET.parse(urdf_path).getroot()
joints = {j.get("name"): j for j in root.findall("joint")}
children = {}
for j in root.findall("joint"):
    children.setdefault(j.find("parent").get("link"), []).append(j)
base = [l.get("name") for l in root.findall("link") if l.get("name") not in {j.find("child").get("link") for j in root.findall("joint")}][0]

# ---- joint order: the URDF's depth-first order over the moving joints
order = []
def walk(link):
    for j in children.get(link, []):
        if j.get("type") == "revolute":
            order.append(j.get("name"))
        walk(j.find("child").get("link"))
walk(base)
leg_order = [n for n in order if re.fullmatch(r".*_(%s)" % "|".join(LEG_JOINTS), n)]
head_order = [n for n in order if n in HEAD_JOINTS]
assert len(order) == 14 and len(leg_order) == 10 and len(head_order) == 4, (order, leg_order, head_order)

# ---- the MuJoCo model
text = open(urdf_path).read()
text = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" balanceinertia="true" discardvisual="false" fusestatic="false"/></mujoco>' % assets_parent.replace("\\", "/"), text, count=1)
spec = mujoco.MjSpec.from_string(text)
spec.body(base).add_freejoint()
spec.option.timestep = SIM_DT
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.gravity = [0, 0, -9.81]
spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], friction=[1.0, 0.005, 0.0001], contype=0, conaffinity=1)
for g in spec.geoms:
    if g.name != "floor":
        g.friction = [1.0, 0.005, 0.0001]
        if g.contype or g.conaffinity:      # the robot's collision geoms hit the floor and not each other (self collision is off in training)
            g.contype, g.conaffinity = 1, 0
for j in spec.joints:
    if j.type == mujoco.mjtJoint.mjJNT_HINGE:
        j.armature, j.damping, j.frictionloss = ARMATURE, [0.0, 0.0, 0.0], 0.0
        a = spec.add_actuator(name=j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0] = KP
        a.biasprm[:3] = [0.0, -KP, -KD]
model = spec.compile()
xml = spec.to_xml()
xml = re.sub(r'meshdir="[^"]*"', 'meshdir="."', xml)
if 'meshdir=' not in xml:
    xml = xml.replace("<compiler", '<compiler meshdir="."', 1)
open(os.path.join(out, "microduck.xml"), "w").write(xml)
used = sorted(set(re.findall(r'file="([^"]+)"', xml)))
for f in used:
    src = os.path.join(assets_parent, f)
    dst = os.path.join(out, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
print("model: %d bodies, %d joints, %d actuators, %.3f kg; %d mesh files copied" % (model.nbody, model.njnt, model.nu, model.body_mass.sum(), len(used)))

# ---- ofxRobot descriptors: one serial chain per limb, from the trunk
def vec(s, d=(0.0, 0.0, 0.0)): return tuple(float(v) for v in s.split()) if s else d
def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
def qaxis(axis, t):
    s = math.sin(t / 2); return (math.cos(t / 2), s*axis[0], s*axis[1], s*axis[2])
def qrpy(r, p, y): return qmul(qmul(qaxis((0, 0, 1), y), qaxis((0, 1, 0), p)), qaxis((1, 0, 0), r))   # URDF: Rz(y) Ry(p) Rx(r)
def qfromto(u, v):
    d = sum(a*b for a, b in zip(u, v))
    if d < -0.999999:
        perp = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (0.0, 1.0, 0.0)
        c = (u[1]*perp[2]-u[2]*perp[1], u[2]*perp[0]-u[0]*perp[2], u[0]*perp[1]-u[1]*perp[0]); n = math.sqrt(sum(x*x for x in c))
        return (0.0, c[0]/n, c[1]/n, c[2]/n)
    c = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]); w = 1.0 + d; n = math.sqrt(w*w + sum(x*x for x in c))
    return (w/n, c[0]/n, c[1]/n, c[2]/n)

def limb(name, joint_names, tool_joint):
    jd, link_names = [], [base]
    for jn in joint_names:
        j = joints[jn]
        o = j.find("origin")
        xyz = vec(o.get("xyz") if o is not None else None); rpy = vec(o.get("rpy") if o is not None else None)
        ax = vec(j.find("axis").get("xyz"), (1.0, 0.0, 0.0)); n = math.sqrt(sum(v*v for v in ax)); ax = tuple(v/n for v in ax)
        raxis = qfromto((1.0, 0.0, 0.0), ax)                  # local X onto the joint axis
        rot0 = qmul(qrpy(*rpy), raxis)
        lim = j.find("limit")
        jd.append({"name": jn, "link": j.find("child").get("link"), "origin": [round(v, 7) for v in xyz],
                   "rot0_wxyz": [round(v, 9) for v in rot0], "rot1_wxyz": [round(v, 9) for v in raxis],
                   "limits_deg": [round(math.degrees(float(lim.get("lower"))), 4), round(math.degrees(float(lim.get("upper"))), 4)]})
        link_names.append(j.find("child").get("link"))
    tj = joints[tool_joint]; o = tj.find("origin")
    desc = {"name": "microduck " + name, "units": {"length": "m", "angle": "deg"}, "upAxis": "Z", "baseLink": base,
            "linkNames": link_names, "joints": jd,
            "tool": {"link": tj.find("child").get("link"), "origin": list(vec(o.get("xyz"))), "rpy": list(vec(o.get("rpy")))},
            "notes": ["Written by tools/make_microduck.py from microduck.urdf: one limb as a serial chain from the trunk.",
                      "Per joint: T(theta) = translate(origin) * R(rot0) * Rx(theta) * R(rot1)^-1",
                      "tool: the fixed frame after the last joint (the foot, or the mouth tip), in that joint's link frame."]}
    json.dump(desc, open(os.path.join(out, name + ".joints.json"), "w"), indent=2)

limb("left_leg", ["left_" + s for s in LEG_JOINTS], "left_foot_joint")
limb("right_leg", ["right_" + s for s in LEG_JOINTS], "right_foot_joint")
limb("head", HEAD_JOINTS, "mouth_tip_joint")

# ---- metadata
meta = {
    "joint_order": order,                                   # the observation's joint order (14)
    "action_order": leg_order + head_order,                 # the policy output's order: legs, then head (14)
    "action_scale": [LEG_SCALE] * len(leg_order) + [HEAD_SCALE] * len(head_order),
    "default_pos": {n: DEFAULT_POS.get(n, 0.0) for n in order},
    "init_height": INIT_HEIGHT, "kp": KP, "kd": KD, "sim_dt": SIM_DT, "decimation": DECIMATION,
    "command_range": COMMAND_RANGE, "trunk_body": base,
    "observation": ["base_lin_vel(3, body)", "base_ang_vel(3, body)", "projected_gravity(3)", "velocity_command(3: vx vy wz)",
                    "joint_pos - default(14)", "joint_vel(14)", "last_action(14)", "look_at_target(3, zeros)"],
    "limbs": {"left_leg": "left_leg.joints.json", "right_leg": "right_leg.joints.json", "head": "head.joints.json"},
}
json.dump(meta, open(os.path.join(out, "microduck.json"), "w"), indent=2)

# The policy as one self-contained file. PyTorch's exporter writes the weights to a sidecar (policy.onnx.data) whose
# name is recorded inside the model, so renaming the pair breaks loading; folding the weights in avoids that. (Needs onnx.)
import onnx
m = onnx.load(policy)                       # reads the sidecar from the same folder
onnx.save(m, os.path.join(out, "walker.onnx"), save_as_external_data=False)
print("walker.onnx: %d bytes, weights embedded" % os.path.getsize(os.path.join(out, "walker.onnx")))
print("wrote", out)
print("joint order:", order)
print("action order:", leg_order + head_order)
