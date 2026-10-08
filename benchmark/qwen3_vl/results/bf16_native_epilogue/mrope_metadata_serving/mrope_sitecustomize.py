import os
import runpy

if os.environ.get("QVL_NATIVE_OUT"):
    runpy.run_path("/root/qvl/experiments/mrope_serving_hook.py")
