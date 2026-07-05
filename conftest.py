"""pytest 根配置:保证未 pip install 时 `import pharos`(及引擎包 chunker/embedder/generator)
可解析。src-layout 过渡态 —— W0b pip install -e .[dev] 落地后此文件收缩/删除。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
