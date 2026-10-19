"""Write ofxRobot limb descriptors for the ANYmal-D: one serial chain per leg (HAA, HFE, KFE), from the trunk, with the foot as the tool.

    python make_limbs.py [<data dir>]        default: ../bin/data/anymal_d

Reads anymal_d.xml with MuJoCo (pip install mujoco), writes LF.joints.json, LH.joints.json, RF.joints.json, RH.joints.json next to it,
and adds "limb_files" and "command_range" to anymal_d.json. The format is ofxRobot's RobotKinematics descriptor (see
ofxRobot/data/*/*.joints.json and example_microduck/tools/make_microduck.py): per joint

    T(theta) = translate(origin) * R(rot0) * Rx(theta) * R(rot1)^-1

which for a MuJoCo body (pos p, quat q, hinge axis a in the body frame, joint at the body origin) is
origin = p, rot1 = rotation taking x onto a, rot0 = q * rot1. The tool is the foot body, a fixed child of the shank.
"""
import json, math, os, sys
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "bin", "data", "anymal_d")
meta_path = os.path.join(out, "anymal_d.json")
meta = json.load(open(meta_path))
m = mujoco.MjModel.from_xml_path(os.path.join(out, "anymal_d.xml"))


def wxyz(r):
    q = r.as_quat()          # x, y, z, w
    return [q[3], q[0], q[1], q[2]]


def from_to(u, v):
    """rotation taking unit vector u onto v"""
    u = np.asarray(u, float); v = np.asarray(v, float)
    d = float(np.dot(u, v))
    if d > 1 - 1e-12:
        return R.identity()
    if d < -1 + 1e-12:
        perp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        return R.from_rotvec(axis / np.linalg.norm(axis) * math.pi)
    axis = np.cross(u, v)
    return R.from_rotvec(axis / np.linalg.norm(axis) * math.acos(d))


def body_quat(b):
    q = m.body_quat[b]       # w, x, y, z
    return R.from_quat([q[1], q[2], q[3], q[0]])


def r6(v):
    return [round(float(x), 7) for x in v]


for leg, joints in meta["limbs"].items():
    jd, links = [], ["base"]
    for jn in joints:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        b = m.jnt_bodyid[j]
        assert np.allclose(m.jnt_pos[j], 0.0), f"{jn}: the joint is not at its body origin"
        axis = m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j])
        rot1 = from_to([1.0, 0.0, 0.0], axis)
        rot0 = body_quat(b) * rot1
        lo, hi = (np.degrees(m.jnt_range[j]) if m.jnt_limited[j] else (-360.0, 360.0))
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        jd.append({"name": jn, "link": name, "origin": r6(m.body_pos[b]),
                   "rot0_wxyz": [round(x, 9) for x in wxyz(rot0)], "rot1_wxyz": [round(x, 9) for x in wxyz(rot1)],
                   "limits_deg": [round(float(lo), 4), round(float(hi), 4)]})
        links.append(name)
    # the tool: the foot body, a fixed child of the last joint's body
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_FOOT")
    assert m.body_parentid[foot] == m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joints[-1])]
    rpy = body_quat(foot).as_euler("xyz")        # extrinsic x, y, z = Rz(y) Ry(p) Rx(r): the URDF / glm convention ofxRobot reads
    desc = {"name": f"ANYmal-D {leg}", "units": {"length": "m", "angle": "deg"}, "upAxis": "Z", "baseLink": "base",
            "linkNames": links, "joints": jd,
            "tool": {"link": f"{leg}_FOOT", "origin": r6(m.body_pos[foot]), "rpy": r6(rpy)},
            "notes": ["Written by tools/make_limbs.py from anymal_d.xml: one leg as a serial chain from the trunk (base).",
                      "Per joint: T(theta) = translate(origin) * R(rot0) * Rx(theta) * R(rot1)^-1",
                      "tool: the foot body, fixed after the last joint, in that joint's link frame."]}
    json.dump(desc, open(os.path.join(out, f"{leg}.joints.json"), "w"), indent=2)
    print("wrote", f"{leg}.joints.json", [j["name"] for j in jd])

meta["limb_files"] = {leg: f"{leg}.joints.json" for leg in meta["limbs"]}
meta["command_range"] = {"vx": [-1.0, 1.0], "vy": [-1.0, 1.0], "wz": [-1.0, 1.0]}   # Isaac Lab velocity_env_cfg.py: trained on [-1, 1] each
json.dump(meta, open(meta_path, "w"), indent=1)
