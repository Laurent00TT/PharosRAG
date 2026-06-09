import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available(), "| n_devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {round(p.total_memory/1024**3,1)}GB sm{p.major}{p.minor}")
# inductor canary (deployment-guide §3.5: torch 2.11 cu13 'duplicate template name' bug)
try:
    import torch._inductor.select_algorithm  # noqa
    print("INDUCTOR_IMPORT_OK")
except Exception as e:
    print("INDUCTOR_IMPORT_FAIL:", type(e).__name__, str(e)[:200])
import vllm, transformers
print("vllm", vllm.__version__, "| transformers", transformers.__version__)
import importlib.metadata as md
for p in ("compressed-tensors", "huggingface-hub"):
    try:
        print(" ", p, md.version(p))
    except Exception as e:
        print(" ", p, "ERR", e)
