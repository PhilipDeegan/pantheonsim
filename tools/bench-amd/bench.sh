#!/usr/bin/env bash
# AMD-engine benchmark harness: see tools/bench-amd/bench.py (--help) and docs/perf-design/rounds/03-amd-engine-baseline.md ("The AMD engine").
#   bench.sh --build build                        time every workload on the simulated MI300X and RX 6800
#   bench.sh --build buildA --build buildB        alternating A/B
#   bench.sh --check-counters --build build       instruction COUNTS against tools/bench-amd/expected (no timing)
exec python3 -I "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/bench.py" "$@"
