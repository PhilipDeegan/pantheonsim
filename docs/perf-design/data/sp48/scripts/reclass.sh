#!/bin/bash
# reclass.sh <prof dir> : classify.py again over the kept perf-script files (no run): the text prof.log is made of
d=$1
for f in "$d"/*.ctl1.script; do
  p=$(basename "$f" .ctl1.script); mode=${p##*.}; prog=${p%.*}; prog=${prog/_//}
  comm=$(awk 'NR==1{print $1}' "$f")
  for ctl in 1 0; do
    echo "== $prog $mode ibs_op cnt_ctl=$ctl (1: micro-ops, 0: cycles)"
    c=$(awk '{n[$1]++} END{b="";m=0; for(k in n) if(n[k]>m){m=n[k];b=k} print b}' "$d/$p.ctl$ctl.script")
    python3 "$(dirname "$0")/classify.py" "$d/$p.ctl$ctl.script" --comm "$c" --insn --top 16
  done
  [ -f "$d/$p.stat" ] && grep -v '^#' "$d/$p.stat"
done
