import os

if os.environ.get("QVL_NATIVE_OUT"):
    import runpy

    runpy.run_path("/root/qvl/experiments/native_dynamic_serving_hook.py")
