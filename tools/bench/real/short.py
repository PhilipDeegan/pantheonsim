#!/usr/bin/env python3
"""short.py: the short programs of the real-program corpora (those that finish
in well under a second on the simulator, as most CI programs do), for
start-up work.  Simulator side only; no GPU card is touched.

  short.py survey --build B --corp C [--cpu N]          wall time of every candidate, one line each
  short.py split  --build B --corp C --programs a,b,..  where the wall time of each goes (perf, call chains)
  short.py ab     --build B --corp C --programs a,b,..  --variant name:K=V,K=V ... interleaved medians

`split` runs each program --reps times under `perf record -F <freq> -g -e
cycles` (user and kernel) and puts every sample in one phase by its call
chain; see PHASES.  perf needs kernel.perf_event_paranoid <= 1 or root: with
--sudo the record step is `sudo -n perf record ... -- sudo -n -u <you> ...`
so the program itself still runs as you.
"""
import argparse, collections, getpass, json, os, re, shutil
import statistics, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real

# Rodinia at the sizes of the card/sim runner (run.sh): tens of milliseconds.
RODINIA_SHORT = [("backprop", ["16384"]), ("gaussian", ["-s", "64"]), ("lud", ["-s", "256", "-v"]),
                 ("needle", ["256", "10"]), ("pathfinder", ["1000", "50", "20"]),
                 ("srad_v2", "64 64 0 31 0 31 0.5 2".split()),
                 ("particlefilter_float", "-x 64 -y 64 -z 5 -np 1000".split())]


def candidates(corp):
    out = []
    for corpus, name, cwd, argv, cp in real.manifest(corp, ["polybench", "rodinia", "hecbench", "samples"]):
        if corpus == "rodinia":
            short = dict(RODINIA_SHORT).get(name)
            if short is not None:
                argv = [argv[0]] + short
        out.append((corpus + "/" + name, cwd, argv, cp))
    return out


def base_env(build, extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VGPU_")}
    env.update(VGPU_QUIET="1", VGPU_GPU="nvidia/rtx3060", VGPU_THREADS="1",
               LD_LIBRARY_PATH=os.path.join(build, "shim"))
    env.update(extra)
    return env


def scratch_for(argv):
    d = tempfile.mkdtemp(prefix="short-")
    dd = os.path.join(os.path.dirname(argv[0]), "data")
    with open(os.path.join(d, "filelist"), "w") as f:
        f.write("".join(os.path.join(dd, "cane4_%d.db\n" % i) for i in range(4)))
    return d


def run(cmd, cwd, env, cp, cpu, timeout):
    sc = scratch_for(cmd) if cp else None
    t0 = time.monotonic()
    try:
        p = subprocess.run(["taskset", "-c", str(cpu)] + cmd, cwd=sc or cwd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired as e:
        rc, out = -999, e.stdout or b""
    wall = time.monotonic() - t0
    extra = b""
    if sc:
        for f in sorted(x for x in os.listdir(sc) if x not in ("filelist", "output.dat")):
            fp = os.path.join(sc, f)
            if os.path.isfile(fp):
                extra += f.encode() + open(fp, "rb").read()
        shutil.rmtree(sc, ignore_errors=True)
    h, _ = real.normalise(out.decode("utf-8", "replace"), extra)
    return wall, rc, h


def wait_load(mx):
    while os.getloadavg()[0] > mx:
        time.sleep(5)


# ---- split -----------------------------------------------------------------
# A sample belongs to the first phase whose pattern matches any frame of its
# call chain, in this order (so decoding done from inside a launch is decode,
# not execution).
PHASES = [
    ("teardown", re.compile(r"^(exit|_exit|__run_exit_handlers|_dl_fini|__cxa_finalize|do_exit|do_group_exit|"
                            r"__x64_sys_exit_group|exit_mmap|__libc_freeres)$|Runtime::~Runtime|Device::~Device|"
                            r"MemoryManager::~MemoryManager|cudaDeviceReset|cuCtxDestroy|UnregisterFatBinary")),
    ("ptx-parse", re.compile(r"vgpu::ptx::|split_ptx|lazy_ptx")),
    ("sass-decode", re.compile(r"sass::decode|annotate_instr|sass::unsupported|okcache|Code::instr|sass::operand|"
                               r"sass::to_text")),
    ("tier-build", re.compile(r"tier::analyze|tier::build|tier::Program|tier::Builder|tier::program_for|"
                              r"tier::src|tier::form_impl|jit::")),
    ("module-load", re.compile(r"sass::Module::load|sass::load|cubin|[Ff]atbin|RegisterFatBinary|cuModuleLoad|"
                               r"Module::index_code|load_module|register_module|elf::|inflate|lz4|decompress")),
    ("execute", re.compile(r"Runner::|step_warp|tier_run|run_grid|run_block|Interpreter::|cudaLaunchKernel|"
                           r"cuLaunchKernel|launch_kernel|Runtime::launch|Device::launch|Device::run_kernel|"
                           r"sass::launch|exec::launch")),
    ("device-init", re.compile(r"cuInit|cuCtxCreate|cudaInitDevice|Runtime::Runtime|runtime::owner|"
                               r"DeviceProfile|profile::|load_profile|Device::Device|MemoryManager::MemoryManager|"
                               r"shared_runtime|visible_devices|ensure_init|ensure_context|telemetry::|"
                               r"FaultHook::FaultHook|load_gpu|yamlish::|backing::")),
    ("loader", re.compile(r"^(_dl_|dl_|_start$|elf_|do_lookup|__GI__dl|_rtld|call_init|"
                          r"__x64_sys_execve|do_execveat|bprm_execve|load_elf_binary|begin_new_exec|setup_arg_pages)"
                          r"|^_dl|rtld")),
]
# Call chains are cut short below the tier's handlers (they are reached through a table of pointers and some keep no
# frame), so a sample no chain rule placed is execution when its leaf is one of the executor's own leaves.
EXEC_LEAF = re.compile(r"simd::|sass::\{anon\}::(h_|tier_|count_|Runner)|exec::memcount|LaunchStats|exec::sasslanes|"
                       r"MemoryManager::(load|store|batch|resolve|run_allocation|find)|\{anon\}::run_(in|out)|"
                       r"vgpu::exec::")


def fname(sym):
    """The function's own name: no parameter list and no template arguments, so that a type named in either (a
    `vgpu::ptx::EntryFn const&` parameter of the launch path) does not place the sample."""
    s = sym.replace("(anonymous namespace)", "{anon}")
    out, depth = [], 0
    for i, ch in enumerate(s):
        if ch == "<" and not s.startswith("operator", max(0, i - 8)):
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif ch == "(" and depth == 0:
            break
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def classify(frames, prog_dso):
    # frames: leaf first, (sym, dso)
    frames = [(fname(s), d) for s, d in frames]
    for name, pat in PHASES:
        for sym, dso in frames:
            if pat.search(sym):
                return name
    if EXEC_LEAF.search(frames[0][0]):
        return "execute"
    for sym, dso in frames:
        if "ld-linux" in dso:
            return "loader"
    user = [(s, d) for s, d in frames if not d.startswith("[")]
    if not user:
        return "kernel-other"
    if any("libcuda" in d or "libvgpu" in d or "libcudart" in d for s, d in user):
        return "sim-other"
    return "program-host"


def perf_split(name, cwd, argv, cp, env, a):
    data = os.path.join(a.tmp, "split-%s.data" % name.replace("/", "_"))
    if os.path.exists(data):
        os.remove(data)
    counts = collections.Counter()
    syms = collections.Counter()
    total = 0
    walls = []
    for r in range(a.reps):
        wait_load(a.max_load)
        sc = scratch_for(argv) if cp else None
        envargs = ["%s=%s" % kv for kv in env.items() if kv[0].startswith(("VGPU_", "LD_LIBRARY", "XDG_", "HOME", "PATH"))]
        inner = ["env", "-i"] + envargs + ["taskset", "-c", str(a.cpu)] + argv
        rec = ["perf", "record", "-o", data, "-F", str(a.freq), "-g", "-e", "cycles", "--"]
        if a.sudo:
            cmd = ["sudo", "-n"] + rec + ["sudo", "-n", "-u", getpass.getuser(), "--"] + inner
        else:
            cmd = rec + inner
        t0 = time.monotonic()
        subprocess.run(cmd, cwd=sc or cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        walls.append(time.monotonic() - t0)
        scr = (["sudo", "-n"] if a.sudo else []) + ["perf", "script", "-i", data, "-F", "comm,ip,sym,dso"]
        txt = subprocess.run(scr, capture_output=True, text=True).stdout
        if a.keep:
            import gzip
            with gzip.open(os.path.join(a.keep, "%s.%d.txt.gz" % (name.replace("/", "_"), r)), "wt") as f:
                f.write(txt)
        if sc:
            shutil.rmtree(sc, ignore_errors=True)
        prog = os.path.basename(argv[0])[:15]
        for block in txt.split("\n\n"):
            lines = block.strip("\n").split("\n")
            if len(lines) < 1 or not lines[0].strip():
                continue
            comm = lines[0].strip()
            if comm != prog:        # sudo, env, taskset before the exec
                continue
            frames = []
            for l in lines[1:]:
                m = re.match(r"\s*[0-9a-f]+\s+(.*?)\s+\((.*)\)\s*$", l)
                if m:
                    frames.append((re.sub(r"\+0x[0-9a-f]+$", "", m.group(1)), os.path.basename(m.group(2))))
            if not frames:
                continue
            ph = classify(frames, prog)
            counts[ph] += 1
            syms[(ph, frames[0][0][:90], frames[0][1])] += 1
            total += 1
    for f in (data, data + ".old"):        # perf keeps the previous run's file as <name>.old
        if a.sudo:
            subprocess.run(["sudo", "-n", "rm", "-f", f])
        elif os.path.exists(f):
            os.remove(f)
    return counts, syms, total, statistics.median(walls)


def cmd_split(a, progs):
    order = [p[0] for p in PHASES] + ["sim-other", "program-host", "kernel-other"]
    res = {}
    for name, cwd, argv, cp in progs:
        env = base_env(a.build, dict(kv.split("=", 1) for kv in a.env))
        run(argv, cwd, env, cp, a.cpu, a.timeout)        # warm the page cache and the verdict file
        w = statistics.median(run(argv, cwd, env, cp, a.cpu, a.timeout)[0] for _ in range(5))
        counts, syms, total, _ = perf_split(name, cwd, argv, cp, env, a)
        res[name] = dict(wall_ms=w * 1e3, samples=total, share={k: counts[k] / max(total, 1) for k in order})
        print("%-34s %7.1f ms %6d samples  " % (name, w * 1e3, total) +
              " ".join("%s=%.1f%%" % (k, 100 * counts[k] / max(total, 1)) for k in order if counts[k]), flush=True)
        if a.top:
            for (ph, s, d), n in syms.most_common(a.top):
                print("      %5.1f%% %-13s %-22s %s" % (100 * n / total, ph, d, s))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


# ---- survey / ab -------------------------------------------------------------
def cmd_survey(a, progs):
    for name, cwd, argv, cp in progs:
        env = base_env(a.build, {})
        w0, rc, _ = run(argv, cwd, env, cp, a.cpu, a.timeout)
        if rc == -999:
            print("%-40s TIMEOUT" % name, flush=True)
            continue
        ws = [run(argv, cwd, env, cp, a.cpu, a.timeout)[0] for _ in range(3)]
        print("%-40s %8.1f ms rc=%d" % (name, statistics.median(ws) * 1e3, rc), flush=True)


def cmd_ab(a, progs):
    variants = []
    for v in a.variant:
        n, _, kv = v.partition(":")
        variants.append((n, dict(x.split("=", 1) for x in kv.split(",") if x)))
    res = {}
    for name, cwd, argv, cp in progs:
        walls = {n: [] for n, _ in variants}
        hashes = {n: set() for n, _ in variants}
        run(argv, cwd, base_env(a.build, {}), cp, a.cpu, a.timeout)        # one untimed run: the page cache
        for r in range(a.runs):
            for n, e in variants:
                wait_load(a.max_load)
                e2 = dict(e)
                pre = e2.pop("PRE", None)
                if pre:
                    subprocess.run(pre, shell=True, check=False)
                w, rc, h = run(argv, cwd, base_env(a.build, e2), cp, a.cpu, a.timeout)
                walls[n].append(w)
                hashes[n].add((rc, h))
        med = {n: statistics.median(walls[n]) for n in walls}
        allh = set().union(*hashes.values())
        res[name] = dict(median_ms={n: med[n] * 1e3 for n in med}, runs_ms={n: [x * 1e3 for x in walls[n]] for n in walls},
                         same=len(allh) == 1)
        print("%-34s " % name + " ".join("%s=%.2f" % (n, med[n] * 1e3) for n, _ in variants) +
              ("  same" if len(allh) == 1 else "  DIFFER"), flush=True)
    first = variants[0][0]
    for n, _ in variants[1:]:
        g = real.geomean([res[p]["median_ms"][first] / res[p]["median_ms"][n] for p in res])
        print("geomean %s/%s = %.4f over %d programs" % (first, n, g, len(res)))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


# ---- selftest ------------------------------------------------------------------
# classify() on chains of the shapes perf gives (leaf first), with the phase each belongs to.  CMake runs it as
# lint_short_split_selftest: the split table is only as good as these rules.
def selftest():
    L, K, LD = "libcudart.so.13.0", "[kernel.kallsyms]", "ld-linux-x86-64.so.2"
    run = ("vgpu::sass::(anonymous namespace)::Runner::run()", L)
    launch = ("cudaLaunchKernel", L)
    cases = [
        # decoding and the tier's analysis happen inside the first launch: they are not execution
        ("sass-decode", [("vgpu::sass::decode(vgpu::sass::Word, unsigned long, int)", L),
                         ("vgpu::sass::Code::instr(unsigned long) const", L), run, launch, ("main", "prog")]),
        ("tier-build", [("vgpu::sass::tier::analyze(vgpu::sass::Instr const&, int, vgpu::sass::tier::Program&, "
                         "vgpu::sass::tier::MOp&)", L), run, launch]),
        ("sass-decode", [("vgpu::sass::okcache::key_of(unsigned char const*, unsigned long)", L),
                         ("vgpu::sass::unsupported[abi:cxx11](unsigned char const*, unsigned long)", L),
                         ("__cudaRegisterFatBinary", L)]),
        ("ptx-parse", [("vgpu::ptx::Parser::parse_instruction()", L), ("cuModuleLoadData", L)]),
        ("module-load", [("vgpu::sass::parse_cubin(unsigned char const*, unsigned long)", L),
                         ("vgpu::sass::load(unsigned char const*, unsigned long, vgpu::MemoryManager&, "
                          "vgpu::DeviceProfile const&)", L)]),
        ("execute", [("vgpu::sass::(anonymous namespace)::Runner::step_warp(vgpu::sass::(anonymous namespace)::"
                      "Block&, vgpu::sass::(anonymous namespace)::Warp&)", L), run, launch]),
        # a handler whose chain perf could not walk
        ("execute", [("void vgpu::sass::(anonymous namespace)::h_global<false>(void*, vgpu::sass::tier::MOp const&, "
                      "unsigned int)", L)]),
        ("execute", [("vgpu::simd::avx2_impl::imad_k(unsigned int*, unsigned int const*, unsigned int const*, "
                      "unsigned int const*)", L)]),
        ("execute", [("vgpu::MemoryManager::load_batch(unsigned long const*, unsigned long, unsigned int, "
                      "unsigned char*)", L)]),
        # a destructor frame inside a launch is not teardown
        ("execute", [("std::lock_guard<std::mutex>::~lock_guard()", L), run]),
        ("teardown", [("vgpu::MemoryManager::run_allocation(unsigned long, unsigned long, unsigned int)", L),
                      ("vgpu::runtime::FaultHook::~FaultHook()", L), ("vgpu::runtime::Device::~Device()", L),
                      ("__run_exit_handlers", "libc.so.6")]),
        ("teardown", [("exit_mmap", K), ("do_exit", K)]),
        ("device-init", [("vgpu::yamlish::parse(std::__cxx11::basic_string<char> const&)", L),
                         ("vgpu::load_gpu(std::__cxx11::basic_string<char> const&)", L)]),
        ("device-init", [("open64", "libc.so.6"), ("vgpu::telemetry::Publisher::Publisher()", L)]),
        ("loader", [("[unknown]", LD), ("[unknown]", LD)]),
        ("loader", [("filemap_map_pages", K), ("_dl_relocate_object", LD)]),
        ("loader", [("load_elf_binary", K), ("bprm_execve", K)]),
        # the kernel working for the program's own code, and a page fault of the program
        ("program-host", [("clear_page_erms", K), ("main", "vectorAdd")]),
        ("program-host", [("random", "libc.so.6"), ("rand", "libc.so.6"), ("main", "vectorAdd")]),
        ("sim-other", [("[unknown]", "libc.so.6"),
                       ("cudaMemcpy_body(void*, void const*, unsigned long, cudaMemcpyKind)", L), ("main", "prog")]),
        ("kernel-other", [("finish_task_switch.isra.0", K), ("schedule", K)]),
        # a ptx:: type among the parameters or template arguments is not PTX parsing
        ("execute", [("vgpu::exec::validate_launch(vgpu::ptx::EntryFn const&, vgpu::exec::LaunchConfig const&, "
                      "vgpu::DeviceProfile const&)", L),
                     ("vgpu::runtime::Device::launch(vgpu::ptx::EntryFn const&, vgpu::exec::LaunchConfig const&)", L)]),
        ("sim-other", [("std::vector<vgpu::ptx::Instr, std::allocator<vgpu::ptx::Instr> >::~vector()", L),
                       ("cudaFree", L)]),
        ("ptx-parse", [("vgpu::ptx::(anonymous namespace)::Parser::parse_instruction()", L),
                       ("vgpu::ptx::parse(std::__cxx11::basic_string<char> const&)", L), launch]),
        ("execute", [("vgpu::exec::(anonymous namespace)::Interpreter::exec_ld(vgpu::ptx::Instr const&)", L)]),
    ]
    bad = 0
    for want, frames in cases:
        got = classify(frames, "prog")
        if got != want:
            bad += 1
            print("selftest: %s classified as %s, expected %s" % (frames[0][0][:60], got, want))
    names = [p[0] for p in PHASES]
    if len(set(names)) != len(names):
        bad += 1
        print("selftest: a phase is listed twice")
    print("selftest: %d cases, %d wrong" % (len(cases), bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["survey", "split", "ab", "selftest"])
    ap.add_argument("--build")
    ap.add_argument("--corp")
    ap.add_argument("--programs", default="")
    ap.add_argument("--cpu", default="12")
    ap.add_argument("--timeout", type=float, default=8)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--freq", type=int, default=20000)
    ap.add_argument("--max-load", type=float, default=8)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--sudo", action="store_true")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--tmp", default=tempfile.gettempdir())
    ap.add_argument("--keep", help="split: a directory for the perf script text of every run (gzip)")
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.mode == "selftest":
        sys.exit(selftest())
    if not a.build or not a.corp:
        ap.error("--build and --corp are required")
    a.build = os.path.abspath(a.build)
    progs = candidates(os.path.abspath(a.corp))
    if a.programs:
        want = a.programs.split(",")
        progs = [p for p in progs if p[0] in want]
    {"survey": cmd_survey, "split": cmd_split, "ab": cmd_ab}[a.mode](a, progs)


if __name__ == "__main__":
    main()
