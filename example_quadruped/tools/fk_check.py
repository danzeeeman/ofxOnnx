"""Check the MJCF's forward kinematics against Newton's FK for the same USD (random joint angles)."""
import sys, numpy as np, newton, warp as wp, mujoco
usd, xml = sys.argv[1], sys.argv[2]
b = newton.ModelBuilder()
b.add_usd(usd, collapse_fixed_joints=False, enable_self_collisions=False, hide_collision_shapes=True)
m = b.finalize()
rng = np.random.default_rng(1)
q = np.array(m.joint_q.numpy())
n = len(q) - 7
ang = rng.uniform(-0.5, 0.5, n)
q[7:] = ang
q[:7] = [0, 0, 0, 0, 0, 0, 1]
m.joint_q.assign(q)
s = m.state()
newton.eval_fk(m, m.joint_q, m.joint_qd, s)
bq = s.body_q.numpy()
mm = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(mm)
names = [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, mm.njnt)]
order = [l.split("/")[-1] for l, t in zip(b.joint_label, b.joint_type) if t == newton.JointType.REVOLUTE]
assert names == order, (names, order)
d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
d.qpos[7:] = ang
mujoco.mj_kinematics(mm, d)
worst = 0
for i in range(b.body_count):
    nm = b.body_label[i].split("/")[-1]
    pm = d.xpos[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, nm)]
    worst = max(worst, np.abs(pm - bq[i][:3]).max())
print("max body position difference vs Newton FK (m):", worst)
