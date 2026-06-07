import importlib.util as u
for m in ["vllm_flash_attn", "flash_attn", "flashinfer"]:
    print(m, bool(u.find_spec(m)))
