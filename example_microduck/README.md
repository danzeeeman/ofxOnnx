# example_microduck

A trained walking policy making a biped walk. The robot is the **microduck** (14 position-controlled joints, 0.74 kg); the policy was
trained in Isaac Lab (the Walk skill, `Isaac-Skill-Walk-Microduck`, rsl_rl PPO) and exported to ONNX. It reads the sensors of the real robot, not the simulator's state. Three libraries each do one job:

| | does | where |
|---|---|---|
| **MuJoCo** | the physics: bodies, joints, servos (position actuators), the floor | `addons/ofxRobot/libs/mujoco` (vendored, opt-in via `mujoco.props`) |
| **ofxOnnx** | the brain: 136 numbers in, 14 joint targets out, 50 times a second (~35 µs) | this addon |
| **ofxRobot** | the robot as data: each limb (two legs, the head) is a serial chain loaded from a JSON descriptor (`RobotKinematics`), forward kinematics from `ArmIk` gives joints and feet, and a `MotionSafetyGate` per limb checks every set of targets before it reaches the servos | `addons/ofxRobot` |

Needs ofxRobot next to ofxOnnx (`addons/ofxRobot`) for its sources and its vendored MuJoCo.

## Run it

```
build.bat                            (or open example_microduck.vcxproj)
bin\example_microduck.exe            it starts walking straight; up/down/left/right to steer
bin\example_microduck.exe --check 30 no window needed: the walker and the followers for 30 simulated seconds: all upright, each within 1.2 m of the one it follows. Then the single-robot check:
bin\example_microduck.exe --check 15 --solo no window needed: walk 15 simulated seconds, print how it went, exit 0 or 1
bin\example_microduck.exe --check 10 --threads   the same, in real time on the physics and inference threads
bin\example_microduck.exe --check 15 --run       the Run policy: run to a target 3 m ahead, stop within 0.4 m of it
bin\example_microduck.exe --swarm       thirty ducks in one walled arena among obstacles, avoiding each other (see "The swarm")
bin\example_microduck.exe --swarm --check 30   the swarm without a window, for 30 simulated seconds
bin\example_microduck.exe --simplify     draw reduced meshes (about 45k triangles in all instead of ~800k per robot), for a weaker GPU
```

Two more microducks (`--ducks N` for N in all; the default is 3) run the **Follow** policy (`Isaac-Skill-Follow-Microduck`) as a chain: the first keeps about 0.6 m behind the one you drive, the second behind the first, and so on, each facing the one ahead of it. They are separate MuJoCo worlds in one coordinate frame, so they do not collide with each other (they can walk through one another), each has its own physics and inference threads, and each "sees" the robot it follows through that simulated camera; the HUD marks a follower `(lost)` while it has not seen the robot ahead for over a second. `1` / `2` switch the robot you drive between the Walk and the Run policy (the followers keep following it).
Walk: `up/down` walk, `left/right` turn, `, .` step sideways, `space` stops. Run: an orange marker on the floor is the target; hold `up/down/left/right` to move it away from, toward, or to either side of the robot (relative to the way it faces), and `space` puts it 2.5 m ahead again. It runs there, stops, and turns to look at it. `x` shoves it, `r` resets, `p` pauses, `k` shows ofxRobot's kinematics
(each limb's joints, and the foot where its forward kinematics puts it, yellow, next to where MuJoCo has it, red: they coincide),
`g` shows the collision geometry, `f` clears a safety-gate fault.

`--check` passes when it stays up, covers at least half the commanded distance, and ofxRobot's forward kinematics agrees with the
simulation to 0.01 mm. On the data here: upright for the whole run, 0.31-0.35 m/s forward for a 0.3 m/s command, FK error 0.00004 mm.

## The swarm

`--swarm` puts **thirty** microducks in one MuJoCo world, a 4 m square arena closed by 0.3 m walls, with **twenty obstacles** in it (eight
kinds, in turn: the box, post and low wall of the training task, and a fat round column, a tall slim pillar, a long thin barrier, an L-shaped
corner and a thin stick; all taller than the time-of-flight sensor, in a fixed random layout). Every duck runs the same policy, the one trained
for Isaac Lab's enclosed navigation task (`Isaac-Skill-NavigateWalkEnclosed-Microduck`, run `2026-10-08_03-26-33`, `model_10250`): the Walk skill's
136 inputs, trained among obstacles and other ducks with collisions ending the episode. Nothing steers them but the policy: each duck is
commanded to walk forwards at its own pace (0.2-0.4 m/s) with a heading that wanders every 8-16 s, and what keeps it clear of walls, obstacles
and the other ducks is what its **time-of-flight sensor** shows the policy.

The ducks have to share one world for that: a ray from one duck's chest hits another duck's collision shapes, and they really collide. So
`Swarm` (`src/Swarm.*`) is one `mjModel` with thirty ducks in it (`swarm.xml`, names prefixed `d0_`, `d1_`, ...), stepped on one physics thread, with one
inference thread running the policy once per duck (about 40 us each), and the ducks' observations, the 64 rays each above all, built in parallel
(without that the rays took 24 ms of a 20 ms period). The ducks collide with each other, the floor, the walls and the obstacles but never with their own
parts, which takes one contact bit per duck, so thirty is the most (`tools/make_swarm.py`). ofxRobot's `MotionSafetyGate` still checks every
duck's joint targets (three limbs each). A duck that falls over, or whose targets the gate refuses, is stood up again at a free spot in the
arena. `r` scatters them again, `p` pauses, `g` shows the collision shapes. The HUD counts, since the last reset, the periods in which ducks touched each
other, a wall or an obstacle, how many ducks have another duck in their sensor's view, and the closest two trunks have been.

What to expect: they do steer, and 30 ducks in 16 m^2 with 20 obstacles is crowded, so they also bump into each other and the obstacles. In a
30 s headless run, 61 of the 435 pairs touched at some point and the closest trunks got to 7 cm. The simulation runs at about 1.6x real time on one thread.

## The data

`bin/data/microduck/` is made by `tools/make_microduck.py` from the robot's URDF and the trained policy (it needs `pip install mujoco==3.13.0 onnx`,
the MuJoCo version vendored in ofxRobot: a model written by a newer one can use attributes the older library rejects):

- `microduck.xml`: the URDF as a MuJoCo model, with a free joint on the trunk, position actuators (kp 2.0, kd 0.1), armature 0.005, friction 1.0,
  robot geometry that does not collide with itself, and a floor: the physics the policy was trained with
- `left_leg / right_leg / head .joints.json`: ofxRobot descriptors
- `microduck.json`: joint order, default pose, action scales, gains, timing
- `swarm.xml`, `navigator.onnx`: the shared world of the swarm and the policy that drives its ducks (made by `tools/make_swarm.py`, below)
- `follower.onnx`: the Follow policy (run `2026-10-05_07-18-23`): the same 136 inputs, velocity zero, and the target is what a camera on the follower's head would measure of the robot ahead (training's `CameraTarget`): seen only inside a 90 x 60 degree view and 3 m, with 3 % range noise and 1 degree angle noise, 5 % dropped frames, the last estimate held for 1 s in the frame it was measured in, then zeros ("not seen"). So the follower has to keep its head on the leader, and the gaps wander between 0.3 and 1 m instead of holding 0.6
- `runner.onnx`: the Run policy (`Isaac-Skill-Run-Microduck`): the same 136 inputs, with the velocity command zero and the target given as a point on the floor (0.1 m up) in the robot's frame
- `walker.onnx`: the Walk policy, with its weights folded into the one file (PyTorch's exporter writes them to a `.onnx.data` file whose name is recorded
  *inside* the model, so renaming the pair breaks loading)
- `assets/`: the meshes

`swarm.xml` and `navigator.onnx` come from `python tools/make_swarm.py <bin/data> <policy.onnx> [ducks=30] [half_side=2.0] [obstacles=20]`
(it only needs `pip install onnx`, and reads the `microduck.xml` above).

## What it took to make the policy walk here

The network is a tiny MLP; the work is reproducing exactly what it saw in training. Things that were not written down, or were easy to get wrong:

- **Joint order.** The 14 inputs and outputs follow Isaac Lab's articulation order. For this robot that is the URDF's own depth-first order (left leg,
  head, right leg; the action is legs then head). It was found by trying the candidate orders: every other one falls within a second.
- **A free joint on the trunk.** MuJoCo's URDF importer fuses the root link into the world, bolting the robot to the floor, and drops its mass (0.54 of 0.74 kg).
- **The sensors.** The 136 inputs are: body IMU gyro, gravity, accelerometer x 0.1; the 14 joint positions (minus the standing pose), velocities and last
  action; the head IMU's gyro and accelerometer x 0.1 and gravity in the head's frame; an 8x8 time-of-flight grid (60 degree field of view on the trunk, tilted
  up 15 degrees, distance over 1.5 m, ordered vertical-major); the skill one-hot (walk), the velocity command, and a zero target. The IMUs are MuJoCo
  accelerometer and gyro sensors on sites at the IMUs' mounting points (proper acceleration, +g up at rest, like a real one); the time-of-flight
  readings are `mj_ray` against the collision shapes.
- **The robot it was trained on is heavier.** The compute board, camera and sensors the URDF lacks (0.179 kg on the trunk, 0.007 on the head) are added to
  the model, with the inertia scaled and the centre of mass shifted as Isaac Lab does; the standing pose is leaned less to compensate. Every servo is
  limited to 0.96 N m, and every action scales by 0.5.
- **Fresh kinematics.** `mj_step` computes body poses before it integrates, so `xpos/xmat` lag the joint angles by a step. A refresh after each step
  (`mj_kinematics`) keeps the observation, the drawing and the forward-kinematics check on the same state. Without it, ofxRobot's FK appeared to be off by 3 mm.
- **Heading hold.** The policy was trained with a heading command: its turn rate was computed from heading error (stiffness 0.5), never given by a
  joystick. Given a plain zero turn rate it veers (a 1.3 m circle here); with the same controller it walks straight.

A red sphere 2 m ahead at eye level, swaying 5 cm left and right, shows where a look-at target would be. Physics and inference run on their own threads: MuJoCo steps in real time (200 Hz) on one, and every 50 Hz period it hands the observation to the other, which runs the policy and answers, so the targets reach the servos as in training. The main thread takes `Microduck::mutex` to read or draw.

The look-at sphere is only drawn: the Walk skill has no look-at input (the separate Look skill does).

Not done: the other skills (look, waddle, stand), robots colliding with each other, and the policy that distils them, and a hardware path (this policy's inputs are all things the real robot has: IMUs, encoders, a time-of-flight sensor).
