"""Headless check: does the pretrained Isaac Lab ANYmal-D flat-velocity policy walk in MuJoCo?

    python sim_check.py [--order bfs|dfs|bfs_rf|dfs_rf] [--cmd vx vy wz] [--seconds 20] [--xml path] [--no-net]

Physics 200 Hz (timestep 0.005 as in training), policy 50 Hz (decimation 4), the ANYdrive LSTM actuator network applied at
every physics step as Isaac Lab does, DC-motor clipping afterwards.
"""
import argparse, json, os, sys
import numpy as np
import torch
import mujoco

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "data", "anymal_d", "anymal_d.xml"))
ap.add_argument("--ckpt", default="C:/src/en_garde_release/IsaacLab/.pretrained_checkpoints/rsl_rl/https/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-AnymalD_newtonmjwarp_none_rsl_rl.pt")
ap.add_argument("--net", required=True, help="anydrive_3_lstm_jit.pt (Isaac Lab Nucleus ActuatorNets/ANYbotics)")
ap.add_argument("--order", default="dfs")
ap.add_argument("--cmd", type=float, nargs=3, default=[0.5, 0.0, 0.0])
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--com-vel", action="store_true", help="base linear velocity of the centre of mass instead of the body origin")
ap.add_argument("--friction", type=float, default=None)
ap.add_argument("--pd", action="store_true", help="plain PD (kp 25 kd 0.5) instead of the actuator network")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--delay", type=int, default=0, help="physics steps (5 ms each) the new targets arrive late: 0 = at once as in training, 4 = one whole policy period")
ap.add_argument("--randomize", type=int, default=None, help="seed: reset like training (joints x U(0.5,1.5), root velocity U(-0.5,0.5))")
ap.add_argument("--pyramidal", action="store_true", help="MuJoCo default cone/impratio instead of the training solver settings")
ap.add_argument("--dump", default=None, help="write the trajectory (time, base pos, quat, joint q) to this .npz")
a = ap.parse_args()

mjl = mujoco.MjModel.from_xml_path(a.xml)
if a.friction is not None:
    mjl.geom_friction[:, 0] = a.friction
if a.pyramidal:
    mjl.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    mjl.opt.impratio = 1.0
d = mujoco.MjData(mjl)

MJ = [f"{leg}_{j}" for leg in ("LF", "LH", "RF", "RH") for j in ("HAA", "HFE", "KFE")]     # the MJCF actuator/joint order
legs = {"bfs": ("LF", "LH", "RF", "RH"), "bfs_rf": ("LF", "RF", "LH", "RH"), "dfs": ("LF", "LH", "RF", "RH"), "dfs_rf": ("LF", "RF", "LH", "RH")}
if a.order.startswith("bfs"):
    POL = [f"{leg}_{j}" for j in ("HAA", "HFE", "KFE") for leg in legs[a.order]]
else:
    POL = [f"{leg}_{j}" for leg in legs[a.order] for j in ("HAA", "HFE", "KFE")]
perm = [MJ.index(n) for n in POL]            # policy index i is MJCF joint perm[i]

default = {"HAA": 0.0}
def default_pos(name):
    leg, j = name.split("_")
    front = leg[0] == "L" and leg[1] == "F" or leg == "RF"
    if j == "HAA":
        return 0.0
    if j == "HFE":
        return 0.4 if front else -0.4
    return -0.8 if front else 0.8
q0 = np.array([default_pos(n) for n in MJ])

jadr = [mjl.joint(n).qposadr[0] for n in MJ]
vadr = [mjl.joint(n).dofadr[0] for n in MJ]
lim = np.array([mjl.jnt_range[mjl.joint(n).id] if mjl.jnt_limited[mjl.joint(n).id] else [-1e9, 1e9] for n in MJ])

sd = torch.load(a.ckpt, map_location="cpu", weights_only=True)["actor_state_dict"]
actor = torch.nn.Sequential(torch.nn.Linear(48, 128), torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(),
                            torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 12))
actor.load_state_dict({k.replace("mlp.", ""): v for k, v in sd.items() if k.startswith("mlp.")})
actor.eval()
net = torch.jit.load(a.net, map_location="cpu").eval()
h = torch.zeros(2, 12, 8); c = torch.zeros(2, 12, 8)

SAT, EFF, VEL = 120.0, 80.0, 7.5
VEL_AT_LIM = VEL * (1 + EFF / SAT)


def actuator(target, q, qd):
    global h, c
    if a.pd:
        return np.clip(25.0 * (target - q) - 0.5 * qd, -EFF, EFF)
    x = torch.tensor(np.stack([target - q, qd], axis=1), dtype=torch.float32).reshape(12, 1, 2)
    with torch.inference_mode():
        tau, (h, c) = net(x, (h, c))
    tau = tau.numpy().astype(np.float64)
    v = np.clip(qd, -VEL_AT_LIM, VEL_AT_LIM)
    top = np.minimum(SAT * (1 - v / VEL), EFF)
    bot = np.maximum(SAT * (-1 - v / VEL), -EFF)
    return np.clip(tau, bot, top)


def reset(height=0.62):
    mujoco.mj_resetData(mjl, d)
    d.qpos[:3] = [0, 0, height]
    d.qpos[3:7] = [1, 0, 0, 0]
    for k, adr in enumerate(jadr):
        d.qpos[adr] = q0[k]
    if a.randomize is not None:
        rng = np.random.default_rng(a.randomize)
        for k, adr in enumerate(jadr):
            d.qpos[adr] = np.clip(q0[k] * rng.uniform(0.5, 1.5), lim[k, 0], lim[k, 1])
        d.qvel[:6] = rng.uniform(-0.5, 0.5, 6)
    mujoco.mj_forward(mjl, d)


def observe(last_action, cmd):
    R = d.xmat[1].reshape(3, 3)                      # body "base" (id 1) orientation, body -> world
    v = d.qvel[:3].copy()
    if a.com_vel:
        com = R @ mjl.body_ipos[1]
        v = v + np.cross(d.qvel[3:6] if False else R @ d.qvel[3:6], com)
    lin = R.T @ v
    ang = d.qvel[3:6].copy()                         # free joint: angular velocity is in the body frame
    g = R.T @ np.array([0.0, 0.0, -1.0])
    q = np.array([d.qpos[i] for i in jadr]); qd = np.array([d.qvel[i] for i in vadr])
    qr = (q - q0)[perm]; qdr = qd[perm]
    return np.concatenate([lin, ang, g, cmd, qr, qdr, last_action]).astype(np.float32), q, qd


reset()
last = np.zeros(12, dtype=np.float32)
cmd = np.array(a.cmd, dtype=np.float32)
steps = int(a.seconds * 50)
rec = []
fell_at = None
stats = dict(h=[], tilt=[], vx=[], hits=0)
pos0 = d.qpos[:2].copy()
for k in range(steps):
    obs, q, qd = observe(last, cmd)
    with torch.no_grad():
        act = actor(torch.from_numpy(obs)[None])[0].numpy()
    last = act
    new_target = q0.copy()
    for i, m in enumerate(perm):
        new_target[m] = q0[m] + 0.5 * act[i]
    if k == 0:
        target = new_target.copy()
    for sub in range(4):
        if sub == a.delay % 4 and a.delay < 4:
            target = new_target                                   # the targets arrive `delay` steps into the period
        q = np.array([d.qpos[i] for i in jadr]); qd = np.array([d.qvel[i] for i in vadr])
        d.ctrl[:] = actuator(target, q, qd)
        mujoco.mj_step(mjl, d)
    if a.delay >= 4:
        target = new_target                                       # a whole period late (or more: held until the next)
    R = d.xmat[1].reshape(3, 3)
    tilt = float(np.degrees(np.arccos(np.clip(-(R.T @ np.array([0, 0, -1.0]))[2] * -1, -1, 1))))
    tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    stats["h"].append(d.qpos[2]); stats["tilt"].append(tilt)
    stats["vx"].append(float((R.T @ d.qvel[:3])[0]))
    qj = np.array([d.qpos[i] for i in jadr])
    stats["hits"] += int(np.sum((qj < lim[:, 0] + 1e-3) | (qj > lim[:, 1] - 1e-3)))
    rec.append(np.concatenate([[k / 50.0], d.qpos[:7], qj]))
    if d.qpos[2] < 0.3 or tilt > 60:
        fell_at = k / 50.0
        break
secs = len(rec) / 50.0
dist = float(np.linalg.norm(d.qpos[:2] - pos0))
print(f"order={a.order} cmd={a.cmd} {'PD' if a.pd else 'net'} {'com' if a.com_vel else 'origin'} vel: "
      f"{'FELL at %.2f s' % fell_at if fell_at else 'upright for %.1f s' % secs}; "
      f"height mean {np.mean(stats['h']):.3f} max {np.max(stats['h']):.3f}; tilt mean {np.mean(stats['tilt']):.1f} max {np.max(stats['tilt']):.1f} deg; "
      f"forward speed mean {np.mean(stats['vx']):.3f} m/s; distance {dist:.2f} m ({dist / max(secs, 1e-9):.2f} m/s); limit hits {stats['hits']}")
if a.dump:
    np.savez(a.dump, rec=np.array(rec))
