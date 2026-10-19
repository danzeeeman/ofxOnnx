"""Convert the ANYmal-D USD (newton-assets, BSD-3, ANYbotics) into a MuJoCo MJCF plus per-link STL meshes.

    python make_anymal_d.py <anymal_d.usda> ../bin/data/anymal_d

Uses Newton's USD importer only to read the model (bodies, joint frames, inertias, shapes); the MJCF is written by hand:
a free joint on the trunk (nothing fused), the legs as nested bodies, the primitive collision shapes the policy was trained
with, one merged visual STL per link in the link's own frame (data/anymal_d/meshes/visual/<link>.stl).
Needs the Isaac Lab venv (newton, trimesh, scipy).
"""
import os, sys
import numpy as np
import newton
import trimesh
from scipy.spatial.transform import Rotation as R

usd, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, "meshes", "visual"), exist_ok=True)

b = newton.ModelBuilder()
b.add_usd(usd, collapse_fixed_joints=False, enable_self_collisions=False, hide_collision_shapes=False)
G = newton.GeoType
VISIBLE, COLLIDE = 1, 2


def tf(x):  # newton transform (px, py, pz, qx, qy, qz, qw) -> (p, Rotation)
    x = np.array(x, dtype=float)
    return x[:3], R.from_quat(x[3:7])


def compose(a, c):
    return a[0] + a[1].apply(c[0]), a[1] * c[1]


def inverse(a):
    ri = a[1].inv()
    return -ri.apply(a[0]), ri


def name(i):
    return b.body_label[i].split("/")[-1]


def fmt(v):
    return " ".join(f"{x:.7g}" for x in v)


def quat_wxyz(r):
    q = r.as_quat()
    return [q[3], q[0], q[1], q[2]]


children = {i: [] for i in range(b.body_count)}
joint_of = {}
for j in range(b.joint_count):
    p, c = b.joint_parent[j], b.joint_child[j]
    joint_of[c] = j
    if p >= 0:
        children[p].append(c)

# merged visual STL per body, vertices expressed in the body frame
for i in range(b.body_count):
    parts = []
    for s in b.body_shapes[i]:
        if b.shape_type[s] == G.MESH and (b.shape_flags[s] & VISIBLE) and not (b.shape_flags[s] & COLLIDE):
            src = b.shape_source[s]
            v = np.array(src.vertices, dtype=float) * np.array(b.shape_scale[s], dtype=float)
            p, r = tf(b.shape_transform[s])
            parts.append(trimesh.Trimesh(r.apply(v) + p, np.array(src.indices).reshape(-1, 3), process=False))
    if parts:
        trimesh.util.concatenate(parts).export(os.path.join(out, "meshes", "visual", name(i) + ".stl"))

dof_start = list(b.joint_qd_start)


def geom_xml(s):
    t = b.shape_type[s]
    p, r = tf(b.shape_transform[s])
    sc = np.array(b.shape_scale[s], dtype=float)
    pose = f'pos="{fmt(p)}" quat="{fmt(quat_wxyz(r))}"'
    common = 'friction="0.8 0.005 0.0001" condim="3"'
    if t == G.SPHERE:
        return f'<geom type="sphere" size="{sc[0]:.7g}" {pose} {common}/>'
    if t == G.BOX:
        return f'<geom type="box" size="{fmt(sc)}" {pose} {common}/>'
    if t == G.CAPSULE:
        return f'<geom type="capsule" size="{sc[0]:.7g} {sc[1]:.7g}" {pose} {common}/>'
    if t == G.CYLINDER:
        return f'<geom type="cylinder" size="{sc[0]:.7g} {sc[1]:.7g}" {pose} {common}/>'
    raise SystemExit(f"unsupported collision shape {t}")


def body_xml(i, parent_world, indent):
    pad = "  " * indent
    world = tf(b.body_q[i])
    lines = []
    if i == 0:
        lines.append(f'{pad}<body name="{name(i)}" pos="0 0 0.62">')
        lines.append(f'{pad}  <freejoint name="root"/>')
    else:
        j = joint_of[i]
        xp, xc = tf(b.joint_X_p[j]), tf(b.joint_X_c[j])
        rel = compose(xp, inverse(xc))            # child body frame in the parent body frame at q = 0
        lines.append(f'{pad}<body name="{name(i)}" pos="{fmt(rel[0])}" quat="{fmt(quat_wxyz(rel[1]))}">')
        if b.joint_type[j] == newton.JointType.REVOLUTE:
            d = dof_start[j]
            axis = xc[1].apply(np.array(b.joint_axis[d], dtype=float))   # joint axis in the child body frame
            jp = xc[0]
            lo, hi = float(b.joint_limit_lower[d]), float(b.joint_limit_upper[d])
            lim = f' range="{lo:.7g} {hi:.7g}" limited="true"' if abs(lo) < 100 else ""
            lines.append(f'{pad}  <joint name="{b.joint_label[j].split("/")[-1]}" type="hinge" pos="{fmt(jp)}" '
                         f'axis="{fmt(axis)}"{lim} damping="0" armature="0" frictionloss="0"/>')
    com = np.array(b.body_com[i], dtype=float)
    I = np.array(b.body_inertia[i], dtype=float).reshape(3, 3)
    lines.append(f'{pad}  <inertial pos="{fmt(com)}" mass="{float(b.body_mass[i]):.7g}" '
                 f'fullinertia="{I[0,0]:.7g} {I[1,1]:.7g} {I[2,2]:.7g} {I[0,1]:.7g} {I[0,2]:.7g} {I[1,2]:.7g}"/>')
    if os.path.exists(os.path.join(out, "meshes", "visual", name(i) + ".stl")):
        lines.append(f'{pad}  <geom type="mesh" mesh="{name(i)}" contype="0" conaffinity="0" group="1" mass="0"/>')
    for s in b.body_shapes[i]:
        if b.shape_flags[s] & COLLIDE:
            lines.append(f"{pad}  " + geom_xml(s))
    for c in children[i]:
        lines += body_xml(c, world, indent + 1)
    lines.append(f"{pad}</body>")
    return lines


joints = [b.joint_label[j].split("/")[-1] for j in range(b.joint_count) if b.joint_type[j] == newton.JointType.REVOLUTE]
mesh_names = [name(i) for i in range(b.body_count) if os.path.exists(os.path.join(out, "meshes", "visual", name(i) + ".stl"))]
xml = ['<mujoco model="anymal_d">',
       '  <!-- ANYmal-D, from the ANYbotics USD in newton-assets (BSD-3-Clause, see LICENSE). Made by tools/make_anymal_d.py. -->',
       '  <compiler angle="radian" meshdir="meshes/visual" autolimits="true"/>',
       '  <!-- the training physics: Isaac Lab / Newton + MuJoCo Warp, sim.dt 0.005, elliptic cone, impratio 100, newton solver, euler -->',
       '  <option timestep="0.005" gravity="0 0 -9.81" cone="elliptic" impratio="100" solver="Newton" iterations="100" ls_iterations="50" integrator="Euler"/>',
       '  <asset>',
       '    <texture name="grid" type="2d" builtin="checker" width="256" height="256" rgb1="0.35 0.37 0.40" rgb2="0.30 0.32 0.35"/>',
       '    <material name="floor" texture="grid" texrepeat="8 8" reflectance="0.05"/>']
xml += [f'    <mesh name="{m}" file="{m}.stl"/>' for m in mesh_names]
xml += ['  </asset>', '  <worldbody>',
        '    <geom name="floor" type="plane" size="0 0 0.05" material="floor" friction="1 0.005 0.0001" condim="3"/>']
xml += body_xml(0, None, 2)
xml += ['  </worldbody>', '  <actuator>']
xml += [f'    <motor name="{j}" joint="{j}" ctrlrange="-80 80" forcerange="-80 80"/>' for j in joints]
xml += ['  </actuator>', '</mujoco>']
with open(os.path.join(out, "anymal_d.xml"), "w", encoding="utf-8") as f:
    f.write("\n".join(xml) + "\n")
print("wrote", os.path.join(out, "anymal_d.xml"), "joints", joints, "meshes", len(mesh_names))
