"""`python -m pharos <cmd>` 入口(未 pip install 也能跑,path-dep 模式与引擎一致)。"""
from .cli import main

if __name__ == "__main__":
    main()
