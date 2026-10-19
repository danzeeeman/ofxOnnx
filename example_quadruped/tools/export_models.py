"""Export the ANYmal-D policy and the ANYdrive actuator network to ONNX, and write reference vectors for a parity check.

    python export_models.py <rsl_rl checkpoint .pt> <anydrive_3_lstm_jit.pt> <out_dir>

policy.onnx    obs[N,48] -> actions[N,12]        (the actor: 3 x 128 ELU, no observation normalisation)
actuator.onnx  x[N,1,2], h[2,N,8], c[2,N,8] -> torque[N], h_out, c_out   (x = pos error * 2, velocity * 0.25 are inside)
reference.json 20 random observations and the policy's outputs, and a 40-step actuator rollout for 12 joints
"""
import json, sys
import numpy as np
import torch

ckpt, jit_path, out = sys.argv[1], sys.argv[2], sys.argv[3]

# ---- policy
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["actor_state_dict"]
actor = torch.nn.Sequential(
    torch.nn.Linear(48, 128), torch.nn.ELU(), torch.nn.Linear(128, 128), torch.nn.ELU(),
    torch.nn.Linear(128, 128), torch.nn.ELU(), torch.nn.Linear(128, 12))
actor.load_state_dict({k.replace("mlp.", ""): v for k, v in sd.items() if k.startswith("mlp.")})
actor.eval()
torch.onnx.export(actor, torch.zeros(1, 48), f"{out}/policy.onnx", input_names=["obs"], output_names=["actions"],
                  dynamic_axes={"obs": {0: "n"}, "actions": {0: "n"}}, opset_version=17, dynamo=False)

# ---- actuator net: the TorchScript module with the LSTM state made explicit
jit = torch.jit.load(jit_path, map_location="cpu").eval()
w = jit.state_dict()


class Actuator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(2, 8, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(8, 1)
        self.register_buffer("in_scale", w["in_scale"].clone())
        self.register_buffer("out_scale", w["out_scale"].clone())
        self.lstm.load_state_dict({k[5:]: v for k, v in w.items() if k.startswith("lstm.")})
        self.linear.load_state_dict({"weight": w["linear.weight"], "bias": w["linear.bias"]})

    def forward(self, x, h, c):
        y, (h2, c2) = self.lstm(x * self.in_scale, (h, c))
        return (self.linear(y).reshape(-1) * self.out_scale), h2, c2


act = Actuator().eval()
N = 12
torch.onnx.export(act, (torch.zeros(N, 1, 2), torch.zeros(2, N, 8), torch.zeros(2, N, 8)), f"{out}/actuator.onnx",
                  input_names=["x", "h", "c"], output_names=["torque", "h_out", "c_out"],
                  dynamic_axes={"x": {0: "n"}, "h": {1: "n"}, "c": {1: "n"}, "torque": {0: "n"}, "h_out": {1: "n"}, "c_out": {1: "n"}},
                  opset_version=17, dynamo=False)

# ---- parity of my re-implementation against the TorchScript network, over a rollout
g = torch.Generator().manual_seed(0)
h = torch.zeros(2, N, 8); c = torch.zeros(2, N, 8); hj = h.clone(); cj = c.clone()
worst = 0.0
steps = []
for t in range(40):
    x = (torch.randn(N, 1, 2, generator=g) * torch.tensor([0.3, 3.0])).float()
    tj, (hj, cj) = jit(x, (hj, cj))
    ta, h, c = act(x, h, c)
    worst = max(worst, float((tj.reshape(-1) - ta).abs().max()))
    steps.append({"x": x.reshape(N, 2).tolist(), "torque": ta.detach().reshape(N).tolist()})
print("actuator: max |torchscript - reimplementation| over 40 steps x 12 joints =", worst)
assert worst < 1e-5

obs = torch.randn(20, 48, generator=g)
with torch.no_grad():
    act_out = actor(obs)
json.dump({"policy": {"obs": obs.tolist(), "actions": act_out.tolist()},
           "actuator": {"n": N, "steps": steps, "note": "h and c start at zero and carry over; x = [pos error, velocity]"}},
          open(f"{out}/reference.json", "w"))
print("wrote policy.onnx, actuator.onnx, reference.json in", out)

import onnx
for f in ("policy", "actuator"):
    onnx.checker.check_model(onnx.load(f"{out}/{f}.onnx"))
    print(f, "onnx checker ok")
