#!/bin/bash
cd /tmp/sp33-jit/e2ework
cat > jit_gauss.sh <<'EOS'
#!/bin/bash
cd /tmp/sp33-jit/corp/rodinia/bin && exec ./gaussian -s 128 2>&1 | grep -v -i -E "time|ms|sec|vgpu\]\[counters"
EOS
cat > jit_segtree.sh <<'EOS'
#!/bin/bash
cd /tmp/sp33-jit/corp/samples/Samples/2_Concepts_and_Techniques/segmentationTreeThrust && exec ./segmentationTreeThrust 2>&1 | grep -v -i -E "time|ms|sec|vgpu\]\[counters"
EOS
cat > jit_radix.sh <<'EOS'
#!/bin/bash
cd /tmp/sp33-jit/corp/samples/Samples/2_Concepts_and_Techniques/radixSortThrust && exec ./radixSortThrust 2>&1 | grep -v -i -E "time|ms|sec|vgpu\]\[counters"
EOS
chmod +x jit_gauss.sh jit_segtree.sh jit_radix.sh
for p in jit_gauss.sh jit_segtree.sh; do
  export VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH=/tmp/sp33-jit/mut/build/shim VGPU_TIER=1
  a=$(flock /tmp/pantheonsim-builder.lock ./$p | md5sum); b=$(flock /tmp/pantheonsim-builder.lock ./$p | md5sum)
  echo "$p tier twice: $a $b $(./$p | wc -l) lines"
done
