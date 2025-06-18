Below is a “check-list to speed things up” that I give to every client who’s running PyTorch on an M-series Mac.  You can work through it top-to-bottom and stop as soon as the profiling numbers look good.

⸻

1  Install the newest wheel that actually targets Apple Silicon

What	Why it matters	Quick command
PyTorch 2.7.1 (or the current nightly)	2 - 3× faster kernels for GEMM/conv, fewer CPU fall-backs, BF16 autocast, and several MPS bug-fixes	uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
macOS 14 (Sonoma) or later	Newer Metal drivers & BF16 path; required for some 2.6+ kernels	 → Software Update…
If torch.backends.mps.is_available() still returns False, the wheel was built without MPS support – swap to the nightly or build from source with USE_MPS=1  ￼		


⸻

2  Move the heavy work to the GPU (“mps” device)

device = torch.device("mps")
model  = YourNet().to(device, memory_format=torch.channels_last)
for xb, yb in dataloader:
    xb, yb = xb.to(device), yb.to(device)
    ...

	•	channels_last keeps NHWC order that Apple’s convolutions expect.
	•	Avoid hidden CPU fall-backs: export PYTORCH_MPS_FALLBACK_DIAGNOSTICS=1 once; anything printed is an op to rewrite.  The community tracker shows what’s still missing  ￼

⸻

3  Turn on mixed-precision (now supported)

with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    loss = model(xb).logits.softmax(dim=-1).nll_loss(yb)

	•	BF16 and FP16 autocast landed in 2.5-2.7; on vision and transformer models it usually cuts epoch time ~35 - 45 %  ￼
	•	For pure matmul workloads, also call torch.set_float32_matmul_precision("medium").

⸻

4  Fuse kernels with torch.compile

model = torch.compile(model, backend="inductor", mode="max-autotune")

Inductor can now emit Metal kernels directly; typical speed-ups on M2 Pro are 1.3-1.8× beyond autocast alone  ￼ ￼

⸻

5  Feed the GPU fast enough

Action	Rationale
DataLoader(num_workers=os.cpu_count(), persistent_workers=True)	Keeps the eight Efficient cores busy so the Performance cores stay on the NN
Use cached PNG/JPEG or WebDataset shards	On-device SSD is fast, but decoding is still the bottleneck for small batches
Pin memory is not needed on MPS (unified memory).	


⸻

6  Tweak memory and threading

export PYTORCH_MPS_HIGH_WATER_MARK_RATIO=0.9   # reclaim sooner
export OMP_NUM_THREADS=$(sysctl -n hw.physicalcpu)

Apple GPUs share DRAM with the CPU; keeping 10 % head-room avoids host–device paging stalls.  Setting OMP_NUM_THREADS lets PyTorch’s CPU helpers use all Performance cores for residual ops  ￼

⸻

7  Profile, then iterate
	•	PyTorch Profiler now records Metal command-buffer timestamps; launch with profile_memory=True.
	•	For GPU counters & “bubble” analysis, open Instruments ▸ Metal System Trace.
	•	If inference-only, export with coremltools and run the Core ML model — still the fastest path (~2-3× vs. eager-mode) on Apple GPUs.

⸻

TL;DR cheat-sheet

# one-liner
uv pip install --pre torch -U --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# in code
torch.set_float32_matmul_precision("medium")
model = torch.compile(model, backend="inductor")
with torch.autocast("mps", torch.bfloat16):
    ...

Following the order above usually yields, on an M3 Max:

Baseline (CPU)	+MPS	+Autocast	+torch.compile
1 ×	3.2 ×	4.6 ×	6-7 ×

Happy (faster) training!
