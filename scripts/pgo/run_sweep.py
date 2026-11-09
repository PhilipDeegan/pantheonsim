#!/usr/bin/env python3
"""Run nvidia/tests/pytorch/sweep.py so that an instrumented simulator writes its profile on the way out.

sweep.py ends with os._exit(): it skips the interpreter's finalisation on purpose (PyTorch's autograd thread can abort the
process there), and os._exit also skips every atexit handler and library destructor, which is where GCC's profile runtime
(-fprofile-generate) writes the .gcda files. A training run through sweep.py itself therefore leaves NO profile for the
libraries PyTorch loads (cuBLAS, cuDNN, cuFFT, cuSOLVER, ...). This wrapper replaces os._exit with libc's exit(): the
destructors run (the profile is written), Py_FinalizeEx still does not.

  vgpu run --gpu nvidia/rtx5090 --preload python scripts/pgo/run_sweep.py [sweep.py arguments]
"""
import ctypes
import os
import runpy
import sys

_libc = ctypes.CDLL(None)


def _exit(code):
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        _libc.exit(int(code))


os._exit = _exit
_sweep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "nvidia", "tests", "pytorch", "sweep.py")
sys.argv = [_sweep] + sys.argv[1:]
runpy.run_path(_sweep, run_name="__main__")
