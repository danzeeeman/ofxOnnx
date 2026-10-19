# example_quadruped

A trained walking policy making a quadruped walk. The robot is the **ANYmal-D** (12 joints, 52 kg); the policy is Isaac Lab's pretrained
`Isaac-Velocity-Flat-AnymalD` (a 3 x 128 ELU network, trained with Newton + MuJoCo Warp) exported to ONNX, and it is driven the way ofxRobot
drives every robot: as a **streaming client**, so an app that drives an arm through `EgmManager`, `LbrClient`, `UrClient`, `XArmClient` or
`StandardBotsClient` drives this robot the same way (and could swap this class for a real controller's).

Three libraries each do one job, as in `example_microduck`:

| | does | where |
|---|---|---|
| **MuJoCo** | the physics: the robot's bodies, joints and the floor, 5 ms steps | `addons/ofxRobot/libs/mujoco` (vendored, opt-in via `mujoco.props`) |
| **ofxOnnx** | two networks: the **policy** (48 numbers in, 12 joint targets out, 50 times a second, ~45 us) and the **ANYdrive actuator network**, an LSTM that turns position error and joint velocity into torque at every 5 ms step (~90 us for all 12 joints) | this addon |
| **ofxRobot** | the robot as data and as a controller: leg descriptors and forward kinematics, the `RobotState` exchange, the safety gate, drawing, video, publishing (table below) | `addons/ofxRobot` |

Needs ofxRobot and ofxNetwork next to ofxOnnx (`addons/ofxRobot`, `addons/ofxNetwork`).

## Run it

```
build.bat                                (or open example_quadruped.vcxproj; Release x64)
bin\example_quadruped.exe                it stands for a second, streaming turns on, and it walks; arrow keys steer
bin\example_quadruped.exe --parity       both networks through ofxOnnx against PyTorch's answers: PASS/FAIL, exit 0/1
bin\example_quadruped.exe --check 20     no window needed: the contract checks and a 20 s walk: PASS/FAIL, exit 0/1
bin\example_quadruped.exe --check 20 --threads    the same, the walk on the controller's real-time worker thread
bin\example_quadruped.exe --record frames --fps 30 --size 1920x1080 --seconds 30 [--lock]     a scripted walk as PNG frames
bin\example_quadruped.exe --publish      each leg's RobotState as JSON over TCP / UDP (ports from 7001)
bin\example_quadruped.exe --hold         do not turn streaming on by itself: it stands until you press s
bin\example_quadruped.exe --shot x.png [--kin] [--collision]     a screenshot of the window, a few seconds in
```

Keys: `up/down` walk, `left/right` turn, `, .` sidestep, `space` stop, `s` streaming on/off (off: the legs hold the stand pose), `x` shove,
`r` reset, `p` pause, `k` ofxRobot's kinematics (each leg's joints, and the foot where its forward kinematics puts it, yellow, over where MuJoCo
has it, red: they coincide), `g` collision shapes, `f` clear a safety-gate fault (streaming stays off until `s`), drag / wheel orbit and zoom.
The camera follows the robot, damped, three quarters from the front. Dots on the floor are where the feet came down (one colour per diagonal
pair of a trot), rings are the feet that are down now.

`--record` (or `--render`) steps the world by the frame time, not the clock, in whole 5 ms physics steps, so any frame rate gives the same
video on any machine; `--lock` is a fixed camera beside the path instead of the follow camera.

## What it uses of ofxRobot

| class | what it does here |
|---|---|
| `RobotState` | the exchange between the app and the controller. A `JointVec` holds up to 7 joints and the robot has 12, so there is **one `RobotState` per leg** (LF, LH, RF, RH; HAA, HFE, KFE in degrees), the same split as the leg descriptors |
| the streaming-client contract (`AGENTS.md`) | `QuadrupedClient` has the shape of `StandardBotsClient` and its siblings: `setup(legs)`, `connect(config)` (non-blocking, a worker thread), `update()` once per frame under a short lock, `consumeJustConnected()`, `close()`, `setStreaming` (**off by default: the legs hold the stand pose**), `setJointLimits`, a per-cycle speed cap, faults that stop streaming and are never cleared by themselves |
| `MotionSafetyGate` | `observe()` on every leg's command, every 5 ms cycle, in the worker; a NaN or an impossible jump is a fault: streaming stops, the legs go back to the stand pose, and it stays that way until the operator clears it (`f`) and turns streaming on again (`s`) |
| `RobotKinematics` | the four legs, loaded from `LF/LH/RF/RH.joints.json` (robots are data: `tools/make_limbs.py` writes them from the MJCF) |
| `ArmIk` | forward kinematics of each leg (`paramsFromChainFkOnly`, `linkFrames`): the foot position, checked against MuJoCo's (worst **0.0002 mm**), and the `tcp` of each published leg state |
| `RobotMesh` | loads the binary STL of every link (`robotmesh::loadBinaryStl`, `smoothNormals`), drawn at the pose the last physics step published |
| `DomeLight`, `BrandColour` | the lighting (a dome of eight lights, as in `example_trajectory`) and the shell colour from ofxRobot's make colours |
| `FrameRecorder` | `--record` / `--render`: frame-by-frame PNGs at any size, `--seconds`, `--fps`, MSAA |
| `LabelBatch` | the HUD, drawn in two calls (several text draws in one frame can all come out with the last one's geometry on this OF build) |
| `StatePublisher` | `--publish`: one per leg, JSON over TCP and UDP with units in the keys (`current`, `target`, `commanded` in degrees, `tcp_mm` the foot in the trunk frame). A leg has 3 joints and the wire format 6, so the last three are zero |
| not used | `ArmJacobian` (a foot-Jacobian readout would fit the HUD but is not there), `MotionPlan`, `RmpFlow` |

The loop, once per frame (or per physics step in a recording):

```
QuadrupedBrain::tick    reads sense() (the body: velocity, gravity, as an IMU and state estimator would report them) and the
                        legs' joints; when a new 20 ms policy period has begun it asks the policy and writes the 12 targets into
                        the four RobotStates (targetJoints, degrees)
QuadrupedClient::update RobotState -> controller (targets in; currentJoints, commandedJoints, connected, time, counters out)
worker, every 5 ms      streaming on? -> MotionSafetyGate -> joint limits -> speed cap -> ANYdrive actuator network -> motor
                        limits -> mj_step; streaming off -> the stand pose
```

The brain is the app's side (`QuadrupedBrain`, `src/`), the controller is the client's side (`QuadrupedClient`): nothing about the policy
is inside the client, so a real controller's client could take its place.

## The data

`bin/data/anymal_d/`, made by the scripts in `tools/` (they need the Isaac Lab Python environment: `torch`, `onnx`, `mujoco`, `newton`, `trimesh`, `scipy`):

- `anymal_d.xml`: the ANYmal-D as a MuJoCo model, from ANYbotics' USD (`make_anymal_d.py <anymal_d.usda> ../bin/data/anymal_d`): a free joint on the trunk,
  the primitive collision shapes the policy was trained with, one merged visual STL per link (`meshes/visual`), motors, the training solver options
- `LF/LH/RF/RH.joints.json`: the ofxRobot leg descriptors, the foot as the tool (`make_limbs.py`)
- `anymal_d.json`: joint order, default pose, action scale, rates, observation layout, the actuator network's clipping, command ranges
- `policy.onnx`, `actuator.onnx`, `reference.json`: from the checkpoint and the TorchScript actuator network (`export_models.py <checkpoint.pt> <anydrive_3_lstm_jit.pt> ../bin/data/anymal_d`);
  `reference.json` has 20 observations with PyTorch's actions and a 40-step actuator rollout, which `--parity` replays
- `LICENSE`: ANYbotics' BSD-3-Clause, for the model

`tools/sim_check.py` is the Python reference of the same loop (it is how the conventions below were found); `fk_check.py` checks the MJCF's kinematics against Newton's.

## What it took to make the policy walk here

The networks are tiny; the work is reproducing exactly what the policy saw in training.

- **Joint order.** The 12 inputs and outputs are depth-first by leg, LF, LH, RF, RH, each HAA, HFE, KFE: the same as the model's. Found by trying the candidates
  in simulation: leg-by-type order falls within 2-3 s, and so does LF, RF, LH, RH.
- **The actuator is not a PD controller.** It was trained with the ANYdrive LSTM actuator network (position error and joint velocity in, torque out), run at every
  physics step, its state carried, then clipped to the motor's torque-speed curve (limit 80 N m, saturation 120, velocity limit 7.5 rad/s). A plain PD falls in 0.4 s.
  This is the ANYmal-C network, which Isaac Lab itself reuses for the ANYmal-D.
- **The body velocity is in the body frame.** MuJoCo's free joint reports the linear velocity in the world frame (rotate by R^T) and the angular velocity already
  in the body frame.
- **5 ms physics, four steps per policy step (50 Hz)**, with the targets constant over the four. (example_microduck's physics is also 5 ms; its policy period is the same.)
- **Heading hold.** The policy was trained with a heading command: its turn rate came from the heading error (stiffness 0.5), never from a joystick. With no turn
  commanded the app holds the heading the robot had; without it the robot drifts a few degrees off line over ten metres.
- **The policy tolerates latency.** Targets that arrive up to a whole policy period (20 ms) late walk just as well, which is what lets the targets go through
  `RobotState` and the main thread (and the real-time worker) instead of being applied inside the physics step. (`tools/sim_check.py --delay`.)
- **MuJoCo 3.12 to 3.13.** The model was written with a newer MuJoCo than the vendored 3.13.0; it uses only attributes both have.
- **The stand pose sags.** With streaming off the legs hold the default pose through the actuator network, which is softer than a PD: the body stands at 0.55 m, and
  walks at 0.62 m.

## Checks

`--parity`: policy 20 observations, worst difference 5e-7; actuator network 40 steps x 12 joints with the LSTM state carried, worst 2e-5 N m.
`--check 20` (and `--threads`) judges three things:
1. streaming is off by default and the legs hold the stand pose (the command equals it to 1e-7 degrees; it does not move although the brain asks for 0.5 m/s);
2. a NaN target stops streaming with a fault, `setStreaming(true)` is refused while faulted, clearing the fault does not resume streaming, and turning it on again works;
3. the walk: upright for 20 s, 0.47 m/s for a command of 0.5, ofxRobot's forward kinematics agrees with MuJoCo to 0.0002 mm, the safety gate never faults, the
   largest command step (28 degrees in 5 ms) is far below the speed cap.

## Limits

This is a simulation. The policy is untested on a robot, the sim-to-real gap is untouched (the actuator network is the ANYmal-C's, and the physics is MuJoCo's, not the
robot's), and the policy's observation includes the base linear velocity, which a real robot would have to estimate. `QuadrupedClient` has no hardware behind it.
Streaming turns itself on after a second here because nothing can be hurt; an app on hardware would leave that to the operator (`--hold`).

## Credits

The ANYmal-D model is ANYbotics' (BSD-3-Clause, via the newton-assets repository); the policy is Isaac Lab's pretrained `Isaac-Velocity-Flat-AnymalD`
checkpoint (`newtonmjwarp`), and the actuator network is Isaac Lab's `anydrive_3_lstm_jit.pt`.
