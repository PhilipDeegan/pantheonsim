#!/usr/bin/env python3
"""stepmix.py (run in the directory that holds work/step-u.txt and work/step-l.txt, the counts of stepcount.so for the unlinked and the linked side): the
instructions the generated object executed a warp (the unlinked window is 16 warps, the linked one 9), by the kind of operand (objdump of the object) and by mnemonic."""
import collections, re, subprocess, sys
def load(step, norm):
    mod=None; cnt={}
    for l in open(step):
        if l.startswith("#"): continue
        p,off,n=l.rsplit(None,2)
        if re.search(r"/[0-9a-f]{32}\.so$",p): mod=p; cnt[int(off,16)]=int(n)
    dis={}
    for l in subprocess.run(["objdump","-d","--no-show-raw-insn",mod],capture_output=True,text=True).stdout.splitlines():
        m=re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$",l)
        if m: dis[int(m.group(1),16)]=(m.group(2),m.group(3))
    cat=collections.Counter(); mn=collections.Counter()
    for a,n in cnt.items():
        op,args=dis.get(a,("?",""))
        mn[op]+=n
        if "(%rsp" in args: cat["stack operand"]+=n
        elif re.search(r"\(%r", args): cat["other memory operand"]+=n
        elif op.startswith("j") or op in ("call","ret"): cat["branch/call/ret"]+=n
        else: cat["register only"]+=n
    tot=sum(cnt.values())
    return {k:v/norm for k,v in cat.items()}, {k:v/norm for k,v in mn.items()}, tot/norm
cu,mu,tu=load("work/step-u.txt",16.0); cl,ml,tl=load("work/step-l.txt",9.0)
print("generated code, instructions a warp: unlinked %.0f linked %.0f (%+.0f)"%(tu,tl,tl-tu))
for k in sorted(set(cu)|set(cl)): print("  %-22s %9.0f %9.0f %+8.0f"%(k,cu.get(k,0),cl.get(k,0),cl.get(k,0)-cu.get(k,0)))
for k in sorted(set(mu)|set(ml), key=lambda k:-abs(ml.get(k,0)-mu.get(k,0)))[:14]: print("  %-12s %9.0f %9.0f %+8.0f"%(k,mu.get(k,0),ml.get(k,0),ml.get(k,0)-mu.get(k,0)))
