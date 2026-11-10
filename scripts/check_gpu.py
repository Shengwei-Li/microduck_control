"""Check that PyTorch and Warp can use the GPU. This takes a few seconds.

Run it from the upstream folder so that it uses the upstream environment:
    cd ~/projectD/third_party/microduck_rl
    uv run python /mnt/d/EE_study/mphil_ready/projectD_rl_payload_biped/scripts/check_gpu.py
"""

import mujoco
import torch
import warp as wp

print("torch", torch.__version__, "| CUDA build", torch.version.cuda)
print("mujoco", mujoco.__version__, "| warp", wp.__version__)

# 1. PyTorch: the GPU must be visible and its architecture must be supported.
assert torch.cuda.is_available(), "PyTorch does not see a GPU"
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # RTX 50 is (12, 0)
print("architectures in this torch build:", torch.cuda.get_arch_list())

# A small matrix product. An unsupported GPU fails here, not at import.
x = torch.randn(2000, 2000, device="cuda")
y = (x @ x).sum().item()
print("torch matmul on GPU: ok")

# 2. Warp: compile and run one small kernel on the GPU.
wp.init()


@wp.kernel
def add_one(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = a[i] + 1.0


a = wp.zeros(1000, dtype=float, device="cuda:0")
wp.launch(add_one, dim=1000, inputs=[a], device="cuda:0")
assert abs(a.numpy().sum() - 1000.0) < 1e-3
print("warp kernel on GPU: ok")

# 3. Warp and PyTorch must share GPU memory without a copy (mjlab needs this).
t = wp.to_torch(a)
assert t.is_cuda and t.sum().item() == 1000.0
print("warp -> torch zero-copy: ok")

print("memory used by torch:", torch.cuda.memory_allocated() // 2**20, "MiB")
