#!/usr/bin/env python3
"""workload_matrix.py names the machine whose job wrote no report (pantheonsim #545)."""
import pathlib
import subprocess
import sys
import tempfile

here = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    wf = d / "w.yml"
    wf.write_text("matrix:\n  include:\n"
                  "    - { key: a10-x1-cuda12.0, gpu: nvidia/a10, count: 1, cuda: '12.0' }\n"
                  "    - { key: rtx-a5000-x1-cuda12.0, gpu: nvidia/rtx-a5000, count: 1, cuda: '12.0' }\n")
    r = d / "results" / "workloads-a10-x1-cuda12.0"
    r.mkdir(parents=True)
    (r / "results.tsv").write_text("idle\tPASS\t1\t\n")
    (r / "machine.txt").write_text("1 x a10, CUDA 12.0\n")
    out = subprocess.run([sys.executable, str(here / "workload_matrix.py"), str(d / "results"), "2", str(wf)],
                         check=True, capture_output=True, text=True).stdout
    assert "1 machine(s) wrote no results at all" in out, out
    assert "`rtx-a5000-x1-cuda12.0`" in out, out
    assert "`a10-x1-cuda12.0`" not in out, out
print("ok")
