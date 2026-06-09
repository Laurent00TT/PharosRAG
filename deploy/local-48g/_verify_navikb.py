import kb, fastapi, qdrant_client, torch, transformers
print("navikb import OK | torch", torch.__version__, "| transformers", transformers.__version__)
try:
    import FlagEmbedding  # noqa
    print("FlagEmbedding OK")
except Exception as e:
    print("FlagEmbedding:", type(e).__name__, str(e)[:120])
