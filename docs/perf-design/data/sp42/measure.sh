#!/bin/bash
# measure.sh <name> <cmd...>   env: SIMCAP (s), VERIFY=1
# prints: name native_s sim_s host_s rc_n rc_s rc_h out_sim_vs_native out_host_vs_sim summary
H=$HOME/hg; SHIM=$H/build/shim; OUT=$H/out; mkdir -p $OUT
name=$1; shift
f='time\|^[0-9.]*$\|^GPU\|elapsed\|Elapsed\|seconds\|[0-9] ms\|[0-9] us\|GB/s\|MB/s\|Throughput\|throughput\|Bandwidth\|bandwidth\|[0-9]s$'
now() { date +%s.%N; }
w=$(mktemp -d $H/work/r.XXXXXX); cd $w
run() { local tag=$1 cap=$2; shift 2; local a=$(now); env "$@" timeout $cap "${CMD[@]}" > $w/$tag.out 2> $w/$tag.err; local rc=$?; local b=$(now); T=$(echo "$b - $a" | bc); RC=$rc; }
CMD=("$@")
run native 600 LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64; tn=$T; rn=$RC
run sim ${SIMCAP:-900} VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_QUIET=1 LD_LIBRARY_PATH=$SHIM; ts=$T; rs=$RC
run host ${HCAP:-900} VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 LD_LIBRARY_PATH=$SHIM; th=$T; rh=$RC
d() { diff <(grep -v -i "$f" $1) <(grep -v -i "$f" $2) > /dev/null && echo same || echo DIFF; }
c1=$(d native.out sim.out); c2=$(d sim.out host.out)
[ "$rs" = 124 ] && c2="sim-timeout($(d native.out host.out) vs native)"
sm=$(grep "^host_gpu: [0-9]* launches" host.err | sed 's/host_gpu: //; s/ (.*//' | head -1)
mb=$(grep "MiB went" host.err | sed 's/host_gpu: //' | head -1)
ref=$(grep -c "refused" host.err)
ver=""
if [ "${VERIFY:-0}" = 1 ]; then run ver ${VCAP:-150} VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 LD_LIBRARY_PATH=$SHIM; ver=$(grep '^host_gpu_verify:' ver.err | tail -1); fi
printf "%s\t%.2f\t%.2f\t%.2f\trc=%s/%s/%s\tsim-vs-native=%s\thost-vs-sim=%s\t[%s]\t[%s]\trefused_lines=%s\t%s\n" "$name" $tn $ts $th $rn $rs $rh $c1 "$c2" "$sm" "$mb" $ref "$ver" | tee -a $OUT/results.tsv
mkdir -p $OUT/$name; cp native.out sim.out host.out host.err ver.err $OUT/$name/ 2>/dev/null; cp sim.err $OUT/$name/ 2>/dev/null
cd $H; rm -rf $w
