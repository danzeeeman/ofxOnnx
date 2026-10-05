# example_microduck

A trained walking policy making a biped walk. The robot is the **microduck** (14 position-controlled joints, 0.74 kg); the policy was
trained in Isaac Lab (`Isaac-Velocity-Flat-Microduck`, rsl_rl PPO) and exported to ONNX. Three libraries each do one job:

| | does | where |
|---|---|---|
| **MuJoCo** | the physics: bodies, joints, servos (position actuators), the floor | `addons/ofxRobot/libs/mujoco` (vendored, opt-in via `mujoco.props`) |
| **ofxOnnx** | the brain: 57 numbers in, 14 joint targets out, 50 times a second (~35 µs) | this addon |
| **ofxRobot** | the robot as data: each limb (two legs, the head) is a serial chain loaded from a JSON descriptor (`RobotKinematics`), forward kinematics from `ArmIk` gives joints and feet, and a `MotionSafetyGate` per limb checks every set of targets before it reaches the servos | `addons/ofxRobot` |

Needs ofxRobot next to ofxOnnx (`addons/ofxRobot`) for its sources and its vendored MuJoCo.

## Run it

```
build.bat                            (or open example_microduck.vcxproj)
bin\example_microduck.exe            it starts walking a circle; up/down/left/right to steer
bin\example_microduck.exe --check 15 no window needed: walk 15 simulated seconds, print how it went, exit 0 or 1
```

`up/down` walk, `left/right` turn, `, .` step sideways, `space` stops, `x` shoves it, `r` resets, `p` pauses, `k` shows ofxRobot's kinematics
(each limb's joints, and the foot where its forward kinematics puts it, yellow, next to where MuJoCo has it, red: they coincide),
`g` shows the collision geometry, `f` clears a safety-gate fault.

`--check` passes when it stays up, covers at least half the commanded distance, and ofxRobot's forward kinematics agrees with the
simulation to 0.01 mm. On the data here: upright for the whole run, 0.31-0.35 m/s forward for a 0.3 m/s command, FK error 0.00004 mm.

## The data

`bin/data/microduck/` is made by `tools/make_microduck.py` from the robot's URDF and the trained policy (it needs `pip install mujoco==3.13.0 onnx`,
the MuJoCo version vendored in ofxRobot: a model written by a newer one can use attributes the older library rejects):

- `microduck.xml`: the URDF as a MuJoCo model, with a free joint on the trunk, position actuators (kp 2.0, kd 0.1), armature 0.005, friction 1.0,
  robot geometry that does not collide with itself, and a floor: the physics the policy was trained with
- `left_leg / right_leg / head .joints.json`: ofxRobot descriptors
- `microduck.json`: joint order, default pose, action scales, gains, timing
- `walker.onnx`: the policy, with its weights folded into the one file (PyTorch's exporter writes them to a `.onnx.data` file whose name is recorded
  *inside* the model, so renaming the pair breaks loading)
- `assets/`: the meshes

## What it took to make the policy walk here

The network is a tiny MLP; the work is reproducing exactly what it saw in training. Things that were not written down, or were easy to get wrong:

- **Joint order.** The 14 inputs and outputs follow Isaac Lab's articulation order. For this robot that is the URDF's own depth-first order (left leg,
  head, right leg; the action is legs then head). It was found by trying the candidate orders: every other one falls within a second.
- **A free joint on the trunk.** MuJoCo's URDF importer fuses the root link into the world, bolting the robot to the floor, and drops its mass (0.54 of 0.74 kg).
- **Centre-of-mass velocity.** Isaac Lab's `base_lin_vel` is the velocity of the trunk's centre of mass; a free joint's `qvel` is that of the frame origin,
  2.3 cm away. They differ by (angular velocity) x (that offset).
- **Fresh kinematics.** `mj_step` computes body poses before it integrates, so `xpos/xmat` lag the joint angles by a step. A refresh after each step
  (`mj_kinematics`) keeps the observation, the drawing and the forward-kinematics check on the same state. Without it, ofxRobot's FK appeared to be off by 3 mm.
- **Heading hold.** The policy was trained with a heading command: its turn rate was computed from heading error (stiffness 0.5), never given by a
  joystick. Given a plain zero turn rate it veers (a 1.3 m circle here); with the same controller it walks straight.

The look-at input is a point 2 m ahead at eye level that sways 5 cm left and right (the policy was trained with 0.4-1.2 m). The simulation and policy run on their own thread at 50 Hz; the main thread takes `Microduck::mutex` to read or draw.

Not done: the sensor-based `microduck_run` policy (IMU, time-of-flight), which needs
those sensor models reproduced; and a hardware path (that policy needs base velocity, which a real robot has no sensor for).
