"""pytest 根配置:保证未 pip install 时 `import pharos` 可解析(path-dep 模式)。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
