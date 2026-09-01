"""Cost measurements for the accuracy-vs-runtime-vs-complexity trade-off (M1).

Everything the brief asks efficiency.json to carry:
  params_total, params_trainable, gflops@224 (+ the tool that produced it),
  gpu_latency_ms_b1, gpu_throughput_ips_b16, cpu_latency_ms_b1 (+ thread count),
  peak_train_vram_mb, train_wallclock_min, batch_size.

Latency is measured on a copy of the model in eval mode with grad disabled; the
GPU numbers synchronise around every timed iteration.
"""
import copy
import statistics
import time

import torch


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def measure_gflops(model, image_size=224, device="cpu"):
    """Returns (gflops, tool). fvcore first, thop as fallback, None if neither."""
    m = copy.deepcopy(model).to(device).eval()
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis
        import logging
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        fca = FlopCountAnalysis(m, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return fca.total() / 1e9, "fvcore.FlopCountAnalysis (multiply-accumulates counted as 1 FLOP)"
    except Exception as e:
        first = f"fvcore failed: {type(e).__name__}: {e}"
    try:
        from thop import profile
        macs, _ = profile(m, inputs=(x,), verbose=False)
        return macs / 1e9, "thop.profile (MACs); " + first
    except Exception as e:
        return None, f"{first}; thop failed: {type(e).__name__}: {e}"


@torch.no_grad()
def gpu_latency(model, image_size=224, batch=1, iters=500, warmup=50, device="cuda:0"):
    if not torch.cuda.is_available():
        return None
    m = copy.deepcopy(model).to(device).eval()
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        m(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(x)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    del m, x
    torch.cuda.empty_cache()
    return statistics.median(ts)


@torch.no_grad()
def cpu_latency(model, image_size=224, iters=100, warmup=10):
    m = copy.deepcopy(model).to("cpu").eval()
    x = torch.randn(1, 3, image_size, image_size)
    for _ in range(warmup):
        m(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(x)
        ts.append((time.perf_counter() - t0) * 1000.0)
    del m
    return statistics.median(ts), torch.get_num_threads()


def measure_all(model, image_size=224, batch16=16, skip_cpu=False):
    """All cost figures except the two that only training can supply."""
    total, trainable = count_params(model)
    gflops, tool = measure_gflops(model, image_size)
    out = {
        "params_total": int(total),
        "params_trainable": int(trainable),
        "params_trainable_pct": round(100.0 * trainable / total, 4) if total else None,
        "gflops_at_224": round(gflops, 4) if gflops is not None else None,
        "gflops_tool": tool,
    }
    lat1 = gpu_latency(model, image_size, batch=1)
    out["gpu_latency_ms_b1"] = round(lat1, 4) if lat1 is not None else None
    lat16 = gpu_latency(model, image_size, batch=batch16, iters=100, warmup=20)
    out["gpu_batch16_latency_ms"] = round(lat16, 4) if lat16 is not None else None
    out["gpu_throughput_ips_b16"] = round(batch16 / (lat16 / 1000.0), 2) if lat16 else None
    if skip_cpu:
        out["cpu_latency_ms_b1"] = None
        out["cpu_threads"] = torch.get_num_threads()
    else:
        c, threads = cpu_latency(model, image_size)
        out["cpu_latency_ms_b1"] = round(c, 4)
        out["cpu_threads"] = threads
    if torch.cuda.is_available():
        out["gpu_name"] = torch.cuda.get_device_name(0)
    return out
