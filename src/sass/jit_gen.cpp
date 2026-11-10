// What the JIT's generated code does for an op it computes itself (docs/perf-design/round11-jit.md, milestone 2): the body of a tier
// handler (tier_ops.inc), with its operand kinds known, calling the same lane kernels (src/simd/kernels.inc) it calls. Each case below
// names the handler it follows; a change to a handler must be made here too (the differential tests run every form both ways).
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#include "jit.hpp"
#include "tier.hpp"

namespace vgpu::sass::jit {

namespace {

using tier::Form;
using tier::SrcKind;
using tier::Dyn;

constexpr uint16_t kNoReg = 0xffff;
constexpr unsigned kRZ = 255;

// Mutation checks (tests only, compiled in with -DVGPU_JIT_MUTANTS=ON): VGPU_JIT_MUTANT=<form name> makes the code generated for that form
// wrong in one lane (a bit of lane 3 of its result, or of lane 2 of a predicate result); VGPU_JIT_MUTANT=maskless makes every generated
// store write the whole row whatever the executing lanes. A differential test that does not fail with each of them does not test the form.
#ifdef VGPU_JIT_MUTANTS
bool mut(const char* what) {
  const char* m = std::getenv("VGPU_JIT_MUTANT");
  return m && std::strcmp(m, what) == 0;
}
#else
constexpr bool mut(const char*) { return false; }
#endif
thread_local const char* g_form_name = "";
thread_local RowCtx* g_rc = nullptr;

struct W {
  std::string s;
  void f(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    s += buf;
  }
};

bool is_row(const SrcFacts& x) { return x.kind == static_cast<uint8_t>(SrcKind::Row); }
bool is_pool(const SrcFacts& x) { return x.kind == static_cast<uint8_t>(SrcKind::Pool); }
bool is_dyn(const SrcFacts& x, Dyn d) { return x.kind == static_cast<uint8_t>(SrcKind::Dyn) && x.dyn == static_cast<uint8_t>(d); }

// A 32-bit source the generated code can make: a register row, a constant, a register with a modifier, a word of bank 0, a uniform register.
bool src_ok(const SrcFacts& x) {
  return is_row(x) || is_pool(x) || is_dyn(x, Dyn::Reg) || is_dyn(x, Dyn::Bank0) || is_dyn(x, Dyn::UReg);
}
// A 64-bit source (tier_src64): a register pair, a constant pair, bank 0 or uniform registers as a pair.
bool src64_ok(const SrcFacts& x) { return is_row(x) || is_pool(x) || is_dyn(x, Dyn::Bank0) || is_dyn(x, Dyn::UReg); }
// A predicate source (tier_pmask): a per-lane P<n> or a uniform UP<n>.
bool pred_ok(const SrcFacts& x) { return is_dyn(x, Dyn::Reg) || is_dyn(x, Dyn::UReg); }
// A source of the uniform datapath (tier_sc: read in the first executing lane, no modifier): a register row, a constant, a register, bank 0, a uniform register.
bool usrc_ok(const SrcFacts& x) { return is_row(x) || is_pool(x) || is_dyn(x, Dyn::Reg) || is_dyn(x, Dyn::Bank0) || is_dyn(x, Dyn::UReg); }
constexpr uint16_t kNoDst = 0xffff;   // tier::kNoReg

// Declares `const u32* A<k>` for source k (tier_src<Flt>).
void decl_src(W& w, const SrcFacts& x, int k, bool flt) {
  if (is_row(x)) {
    w.f("    const u32* A%d = %s;\n", k, g_rc->row(x.off / 32u, false).c_str());
  } else if (is_pool(x)) {
    w.f("    u32 T%d[32]; splat(T%d, 0x%xu); const u32* A%d = T%d;\n", k, k, x.val, k, k);
  } else if (is_dyn(x, Dyn::Reg)) {
    w.f("    u32 T%d[32]; const u32* A%d = dynreg(T%d, %s, %s, %uu);\n", k, k, k, g_rc->row(x.off, false).c_str(), flt ? "true" : "false", unsigned{x.mod});
  } else if (is_dyn(x, Dyn::Bank0)) {
    w.f("    u32 T%d[32]; splat(T%d, tmod(%s, %uu, b0w(c, %uu))); const u32* A%d = T%d;\n", k, k, flt ? "true" : "false", unsigned{x.mod}, x.off, k, k);
  } else {   // UReg
    w.f("    u32 T%d[32]; splat(T%d, tmod(%s, %uu, c->ur[%u])); const u32* A%d = T%d;\n", k, k, flt ? "true" : "false", unsigned{x.mod}, x.off, k, k);
  }
}

// Declares `const u32* L<k>, *H<k>` for a 64-bit source (tier_src64: no modifier is applied here).
void decl_src64(W& w, const SrcFacts& x, int k) {
  if (is_row(x)) {
    { const std::string lo = g_rc->row(x.off / 32u, false), hi = g_rc->row(x.off / 32u + 1u, false);
      w.f("    const u32* L%d = %s; const u32* H%d = %s;\n", k, lo.c_str(), k, hi.c_str()); }
  } else if (is_pool(x)) {
    w.f("    u32 TL%d[32], TH%d[32]; splat(TL%d, 0x%xu); splat(TH%d, 0x%xu); const u32* L%d = TL%d; const u32* H%d = TH%d;\n", k, k, k, x.val, k, x.val2, k, k, k, k);
  } else if (is_dyn(x, Dyn::Bank0)) {
    w.f("    u32 TL%d[32], TH%d[32]; splat(TL%d, b0w(c, %uu)); splat(TH%d, b0w(c, %uu)); const u32* L%d = TL%d; const u32* H%d = TH%d;\n", k, k, k,
        x.off, k, x.off + 4u, k, k, k, k);
  } else {   // UReg pair
    w.f("    u32 TL%d[32], TH%d[32]; splat(TL%d, c->ur[%u]); splat(TH%d, c->ur[%u]); const u32* L%d = TL%d; const u32* H%d = TH%d;\n", k, k, k, x.off, k, x.off + 1u,
        k, k, k, k);
  }
}

// The value of a uniform-datapath source in lane `l` (tier_sc), as an expression.
std::string usrc(const SrcFacts& x) {
  char buf[128];
  if (is_row(x)) std::snprintf(buf, sizeof buf, "%s[l]", g_rc->row(x.off / 32u, false).c_str());
  else if (is_pool(x)) std::snprintf(buf, sizeof buf, "0x%xu", x.val);
  else if (is_dyn(x, Dyn::Reg)) std::snprintf(buf, sizeof buf, "%s[l]", g_rc->row(x.off, false).c_str());
  else if (is_dyn(x, Dyn::Bank0)) std::snprintf(buf, sizeof buf, "b0w(c, %uu)", x.off);
  else std::snprintf(buf, sizeof buf, "c->ur[%u]", x.off);   // a uniform register
  return buf;
}

const char* pm(W& w, const SrcFacts& x, char* buf, size_t n) {
  const bool neg = x.mod & 1u;
  if (x.dyn == static_cast<uint8_t>(Dyn::UReg)) {   // a uniform predicate: never cached (only an escape writes it)
    std::snprintf(buf, n, "pmask(c, true, %uu, %s)", x.off, neg ? "true" : "false");
  } else {
    std::snprintf(buf, n, "%s", g_rc->pred(x.off, neg).c_str());
  }
  (void)w;
  return buf;
}

void store(W& w, uint16_t dst, const char* row) {
  if (dst == kNoReg) return;
  const std::string target = g_rc->row(dst, true);
  if (mut(g_form_name)) {
    w.f("    { u32 mm[32]; for (unsigned l = 0; l < 32; ++l) mm[l] = %s[l]; mm[3] ^= 1u; store_row(%s, ex, mm); }\n", row, target.c_str());
    return;
  }
  if (mut("maskless")) {
    w.f("    store_row(%s, 0xffffffffu, %s);\n", target.c_str(), row);
    return;
  }
  w.f("    store_row(%s, ex, %s);\n", target.c_str(), row);
}
void setp(W& w, uint16_t reg, const char* v) {
  if (reg == kNoReg) return;
  w.s += g_rc->setp(reg, v);
}
void setp2(W& w, const OpFacts& o, bool negate_second) {   // ISETP / FSETP: the two predicate outputs from r and acc
  char a[96], b[96];
  std::snprintf(a, sizeof a, "combine(%uu, r, acc)", o.b);
  std::snprintf(b, sizeof b, negate_second ? "combine(%uu, ~r, acc)" : "combine(%uu, r, acc)", o.b);
  setp(w, o.dst, a);
  setp(w, o.dst2, b);
}

}  // namespace

// ---- register residence (jit.hpp, RowCtx) ----------------------------------------------------------------------------------

std::string RowCtx::row(unsigned reg, bool write) {
  char buf[64];
  const auto direct = [&] {
    std::snprintf(buf, sizeof buf, "(R + %uu)", reg * 32u);
    return std::string(buf);
  };
  if (!write) {
    reads.insert(reg);
    // A row this region has not written is the register file's own (nothing to copy); one it has written is the local copy.
    const auto it = st.find(reg);
    if (!enabled || it == st.end()) return direct();
    std::snprintf(buf, sizeof buf, "X%u", reg);
    return std::string(buf);
  }
  if (!enabled || (!decl.count(reg) && decl.size() >= cap)) return direct();
  decl.insert(reg);
  char name[16];
  std::snprintf(name, sizeof name, "X%u", reg);
  const auto it = st.find(reg);
  if (it == st.end()) {   // lanes outside the executing mask keep the old value: it must be here first (a whole-row write needs no load)
    if (!mut("res_nomask")) {
      std::snprintf(buf, sizeof buf, "    if (ex != 0xffffffffu) ldrow(%s, R + %uu);\n", name, reg * 32u);
      pre += buf;
    }
    st[reg] = (mut("res_dstsrc") && reads.count(reg)) ? 1 : 2;
  } else if (!(mut("res_dstsrc") && reads.count(reg))) {
    it->second = 2;
  }
  return name;
}

std::string RowCtx::pred(unsigned reg, bool neg) {
  if (reg == 7) return neg ? "0u" : "0xffffffffu";
  char buf[96];
  if (!enabled) {
    std::snprintf(buf, sizeof buf, neg ? "(~c->p[%u])" : "c->p[%u]", reg);
    return buf;
  }
  pdecl.insert(reg);
  if (!pst[reg]) {
    std::snprintf(buf, sizeof buf, "    P%u = c->p[%u];\n", reg, reg);
    pre += buf;
    pst[reg] = 1;
  }
  std::snprintf(buf, sizeof buf, neg ? "(~P%u)" : "P%u", reg);
  return buf;
}

std::string RowCtx::setp(unsigned reg, const std::string& value) {
  if (reg == 7) return {};
  char buf[256];
  if (!enabled) {
    std::snprintf(buf, sizeof buf, "    c->p[%u] = (c->p[%u] & ~ex) | ((%s) & ex);\n", reg, reg, value.c_str());
    return buf;
  }
  pdecl.insert(reg);
  if (!pst[reg]) {
    std::snprintf(buf, sizeof buf, "    P%u = c->p[%u];\n", reg, reg);
    pre += buf;
  }
  pst[reg] = 2;
  std::snprintf(buf, sizeof buf, "    P%u = (P%u & ~ex) | ((%s) & ex);\n", reg, reg, value.c_str());
  return buf;
}

std::string RowCtx::flush_pre() {
  std::string t;
  t.swap(pre);
  return t;
}

void RowCtx::sync(std::string& out, bool invalidate) {
  char buf[96];
  if (!(invalidate && mut("res_nostore_escape")) && !(!invalidate && mut("res_noend"))) {
    for (auto& [reg, state] : st)
      if (state == 2) {
        std::snprintf(buf, sizeof buf, "  strow(R + %uu, X%u);\n", reg * 32u, reg);
        out += buf;
        state = 1;
      }
    for (unsigned r = 0; r < 7; ++r)
      if (pst[r] == 2) {
        std::snprintf(buf, sizeof buf, "  c->p[%u] = P%u;\n", r, r);
        out += buf;
        pst[r] = 1;
      }
  }
  if (invalidate && !mut("res_noreload")) {
    st.clear();
    for (unsigned r = 0; r < 8; ++r) pst[r] = 0;
  }
}

std::string RowCtx::decls() const {
  std::string t;
  char buf[96];
  for (unsigned r : decl) {
    std::snprintf(buf, sizeof buf, "  alignas(32) u32 X%u[32];\n", r);
    t += buf;
  }
  for (unsigned r : pdecl) {
    std::snprintf(buf, sizeof buf, "  u32 P%u;\n", r);
    t += buf;
  }
  return t;
}

bool generates(const OpFacts& o) {
  switch (static_cast<Form>(o.form)) {
    case Form::Mov: return src_ok(o.s[0]);
    case Form::Iadd3: case Form::Imad: case Form::Lop3:
      if (!(src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]))) return false;
      return static_cast<Form>(o.form) != Form::Lop3 || !(o.aux & 1) || (o.aux & 2) || pred_ok(o.p);
    case Form::Iadd3x:
      return src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]) && (!(o.aux & 1) || (pred_ok(o.p) && pred_ok(o.p2)));
    case Form::Imad32: return src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]) && (!(o.aux & 1) || pred_ok(o.p));
    case Form::ImadWide: return src_ok(o.s[0]) && src_ok(o.s[1]) && src64_ok(o.s[2]) && (!(o.aux & 1) || pred_ok(o.p));
    case Form::ImadHi:
      return src_ok(o.s[0]) && src_ok(o.s[1]) && ((o.aux & 4) ? src64_ok(o.s[2]) : src_ok(o.s[2])) && (!(o.aux & 1) || pred_ok(o.p));
    case Form::Lea: return src_ok(o.s[0]) && src_ok(o.s[1]) && ((o.aux & 4) ? src_ok(o.s[2]) : true) && (!(o.aux & 2) || pred_ok(o.p));
    case Form::Shf: return src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]);
    case Form::Isetp: return src_ok(o.s[0]) && src_ok(o.s[1]) && pred_ok(o.p);
    case Form::Fsetp: return o.aux == 0 && src_ok(o.s[0]) && src_ok(o.s[1]) && pred_ok(o.p);
    case Form::Fsel: return src_ok(o.s[0]) && src_ok(o.s[1]) && pred_ok(o.p);
    case Form::Ffma: return o.b == 0 && src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]);
    case Form::Fadd: case Form::Fmul: return o.b == 0 && src_ok(o.s[0]) && src_ok(o.s[1]);
    case Form::Prmt: return o.a == 0 && src_ok(o.s[0]) && src_ok(o.s[1]) && src_ok(o.s[2]);
    case Form::Imnmx: return src_ok(o.s[0]) && src_ok(o.s[1]) && pred_ok(o.p);
    case Form::I2fp: return src_ok(o.s[0]);
    case Form::Sel: return src_ok(o.s[0]) && src_ok(o.s[1]) && pred_ok(o.p);
    case Form::Plop3: return pred_ok(o.s[0]) && pred_ok(o.s[1]) && pred_ok(o.s[2]);
    case Form::S2r: return !(o.a == 0x01 || (o.a >= 0x50 && o.a <= 0x53));   // (the clocks read the warp's step count, which the generated code has not written yet: the handler's)
    case Form::Cs2r: return o.a == 0xff;   // SRZ: zeros (another special register is the handler's)
    case Form::Umov: return usrc_ok(o.s[0]);
    case Form::Uldc: return o.a == 0;   // bank 0 only (the Ctx holds no other bank)
    case Form::Uiadd3:
      return usrc_ok(o.s[0]) && usrc_ok(o.s[1]) && usrc_ok(o.s[2]) && (!(o.aux & 1) || (pred_ok(o.p) && pred_ok(o.p2))) &&
             (o.dst2 == kNoDst || (o.aux & 2)) && (o.a == kNoDst || (o.aux & 4));   // (a predicate result in a lane's own P is the handler's)
    case Form::Ulop3: return usrc_ok(o.s[0]) && usrc_ok(o.s[1]) && usrc_ok(o.s[2]);
    case Form::Usel: return usrc_ok(o.s[0]) && usrc_ok(o.s[1]) && pred_ok(o.p);
    case Form::Uisetp: return usrc_ok(o.s[0]) && usrc_ok(o.s[1]) && pred_ok(o.p);
    default: return false;
  }
}

// ---- fused runs (round sp48; jit.hpp) ---------------------------------------------------------------------------------------

namespace {
// A 32-bit source of a fused op that is the same in every lane and does not change within the run (no fused form writes a uniform register or the
// constant bank): a scalar expression, or "" for a source that is a register row.
std::string fused_scalar(const SrcFacts& x, bool flt) {
  char buf[128] = "";
  if (is_pool(x)) std::snprintf(buf, sizeof buf, "0x%xu", x.val);
  else if (is_dyn(x, Dyn::Bank0)) std::snprintf(buf, sizeof buf, "tmod(%s, %uu, b0w(c, %uu))", flt ? "true" : "false", unsigned{x.mod}, x.off);
  else if (is_dyn(x, Dyn::UReg)) std::snprintf(buf, sizeof buf, "tmod(%s, %uu, c->ur[%u])", flt ? "true" : "false", unsigned{x.mod}, x.off);
  return buf;
}
bool fused_float(Form f) { return f == Form::Ffma || f == Form::Fadd || f == Form::Fmul; }
unsigned fused_nsrc(Form f) { return f == Form::Mov ? 1u : (f == Form::Fadd || f == Form::Fmul || f == Form::Lea) ? 2u : 3u; }
}  // namespace

bool fusable(const OpFacts& o) {
  if (!o.unguarded || o.br || o.cont || o.store || o.form == 0 || !generates(o)) return false;
  switch (static_cast<Form>(o.form)) {
    case Form::Mov: case Form::Iadd3: case Form::Imad: return true;
    case Form::Lop3: return !(o.aux & 1);                                       // (no predicate result)
    case Form::Lea: return !(o.aux & (1 | 2 | 8)) && o.dst2 == kNoReg;          // the plain form: (a << sh) + b, no carry in or out
    case Form::Ffma: case Form::Fadd: case Form::Fmul: return o.a == 0 && o.b == 0;   // (round to nearest, no FTZ: the forms of fadd_k, fmul_k, ffma_k)
    default: return false;
  }
}

std::string gen_fused_run(const OpFacts* ops, uint32_t n, bool full) {
  W w, pre;
  std::set<unsigned> have, written;
  // The value of register row `reg` for the lanes of this block: the variable the run has for it, read from the register file the first time.
  const auto use = [&](unsigned reg) {
    char name[24];
    std::snprintf(name, sizeof name, "g%u", reg);
    if (!have.count(reg) || (mut("fuse_stale") && written.count(reg))) {
      if (have.count(reg)) w.f("      g%u = ld8(R + %uu + q);\n", reg, reg * 32u);
      else w.f("      V8u g%u = ld8(R + %uu + q);\n", reg, reg * 32u);
      have.insert(reg);
    }
    return std::string(name);
  };
  for (uint32_t i = 0; i < n; ++i) {
    const OpFacts& o = ops[i];
    const Form form = static_cast<Form>(o.form);
    const bool flt = fused_float(form);
    std::string a[3];
    for (unsigned k = 0; k < fused_nsrc(form); ++k) {
      const SrcFacts& x = o.s[k];
      char buf[160];
      if (is_row(x)) a[k] = use(x.off / 32u);
      else if (is_dyn(x, Dyn::Reg)) {
        std::snprintf(buf, sizeof buf, "tmod8(%s, %uu, %s)", flt ? "true" : "false", unsigned{x.mod}, use(x.off).c_str());
        a[k] = buf;
      } else {   // the same word in every lane: worked out once, before the loop
        pre.f("    const u32 k%u_%u = %s;\n", i, k, fused_scalar(x, flt).c_str());
        std::snprintf(buf, sizeof buf, "sp8(k%u_%u)", i, k);
        a[k] = buf;
      }
    }
    w.f("      const V8u t%u = ", i);
    switch (form) {
      case Form::Mov: w.f("%s", a[0].c_str()); break;
      case Form::Iadd3: w.f("%s + %s + %s + sp8(%uu)", a[0].c_str(), a[1].c_str(), a[2].c_str(), o.a); break;   // iadd3_k
      case Form::Imad: w.f("%s * %s + %s", a[0].c_str(), a[1].c_str(), a[2].c_str()); break;                     // imad_k
      case Form::Lop3: {   // lop3_k: each set bit t of the table is one minterm of (a, b, c)
        std::string e;
        for (unsigned t = 0; t < 8; ++t)
          if ((o.a >> t) & 1) {
            if (!e.empty()) e += " | ";
            e += std::string("(") + ((t & 4) ? "" : "~") + a[0] + " & " + ((t & 2) ? "" : "~") + a[1] + " & " + ((t & 1) ? "" : "~") + a[2] + ")";
          }
        w.s += e.empty() ? std::string("sp8(0u)") : e;
        break;
      }
      case Form::Lea:   // h_lea, the plain form
        if (o.a >= 32) w.f("%s", a[1].c_str());
        else w.f("(%s << %uu) + %s", a[0].c_str(), o.a, a[1].c_str());
        break;
      case Form::Ffma: w.f("ffma8(%s, %s, %s)", a[0].c_str(), a[1].c_str(), a[2].c_str()); break;   // ffma_k
      case Form::Fadd: w.f("fadd8(%s, %s)", a[0].c_str(), a[1].c_str()); break;                      // fadd_k
      case Form::Fmul: w.f("fmul8(%s, %s)", a[0].c_str(), a[1].c_str()); break;                      // fmul_k
      default: break;
    }
    w.f(";  // op +%u %s\n", i, tier::form_name(form));
    if (o.dst == kNoReg) continue;
    const unsigned d = o.dst;
    char val[48];
    std::snprintf(val, sizeof val, mut((std::string("fuse_") + tier::form_name(form)).c_str()) ? "(t%u ^ sp8(1u))" : "t%u", i);   // (mutants fuse_mov, fuse_iadd3, ...)
    if (full || mut("fuse_noblend")) {   // the whole warp runs it: the row is the result
      w.f(have.count(d) ? "      g%u = %s;\n" : "      V8u g%u = %s;\n", d, val);
      have.insert(d);
    } else {                             // the lanes outside the group keep what the register holds
      const std::string old = use(d);
      w.f("      %s = bl8(m, %s, %s);\n", old.c_str(), val, old.c_str());
    }
    written.insert(d);
  }
  std::string out = pre.s;
  out += "    for (unsigned q = 0; q < 32; q += 8) {\n";
  if (!full) out += "      const V8u m = lm8(ex, q);\n";
  out += w.s;
  unsigned skip = mut("fuse_nostore") && !written.empty() ? *written.rbegin() : ~0u;
  for (unsigned reg : written)
    if (reg != skip) {
      char buf[96];
      std::snprintf(buf, sizeof buf, "      st8(R + %uu + q, g%u);\n", reg * 32u, reg);
      out += buf;
    }
  out += "    }\n";
  return out;
}

std::string gen_op_body(const OpFacts& o, RowCtx& rc) {
  g_rc = &rc;
  rc.reads.clear();
  W w;
  char pa[96], pb[96];
  const Form form = static_cast<Form>(o.form);
  g_form_name = tier::form_name(form);
  const bool flt = form == Form::Fsel || form == Form::Ffma || form == Form::Fadd || form == Form::Fmul || form == Form::Fmnmx;
  w.f("    {\n");
  switch (form) {
    case Form::Mov:
      decl_src(w, o.s[0], 0, false);
      store(w, o.dst, "A0");
      break;
    case Form::Iadd3:   // h_iadd3
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32]; iadd3_k(r, A0, A1, A2, %uu);\n", o.a);
      store(w, o.dst, "r");
      break;
    case Form::Imad:   // h_imad
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32]; imad_k(r, A0, A1, A2);\n");
      store(w, o.dst, "r");
      break;
    case Form::Lop3:   // h_lop3
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32]; lop3_k(r, A0, A1, A2, %uu);\n", o.a);
      if (o.aux & 1) {
        w.f("    u32 pm = static_cast<u32>(icmp_mask_k(%u, false, r, %s, 32));\n", (o.aux & 2) ? 0u : 1u, (o.aux & 2) ? "kOnes32" : "kZeros32");
        if (!(o.aux & 2)) w.f("    pm |= %s;\n", pm(w, o.p, pa, sizeof pa));
        setp(w, o.dst2, "pm");
      }
      store(w, o.dst, "r");
      break;
    case Form::Iadd3x:   // h_iadd3x
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32], over1, over2;\n");
      if (o.aux & 1) w.f("    iadd3x_k(r, A0, A1, A2, %uu, %s, %s, &over1, &over2);\n", o.a, pm(w, o.p, pa, sizeof pa), pm(w, o.p2, pb, sizeof pb));
      else w.f("    iadd3x_k(r, A0, A1, A2, %uu, 0u, 0u, &over1, &over2);\n", o.a);
      store(w, o.dst, "r");
      if (o.dst2 != kNoReg) setp(w, o.dst2, "over1");
      if (o.c != kNoReg) setp(w, static_cast<uint16_t>(o.c), "over2");
      break;
    case Form::Imad32:   // h_imad32
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    const u32 cin = %s;\n    u32 r[32], carry = 0;\n", (o.aux & 1) ? pm(w, o.p, pa, sizeof pa) : "0u");
      w.f("    for (unsigned l = 0; l < 32; ++l) {\n"
          "      const u64 full = static_cast<u64>(A0[l] * A1[l]) + A2[l] + ((cin >> l) & 1u);\n"
          "      r[l] = static_cast<u32>(full);\n"
          "      carry |= static_cast<u32>((full >> 32) != 0) << l;\n    }\n");
      store(w, o.dst, "r");
      if (o.dst2 != kNoReg) setp(w, o.dst2, "carry");
      break;
    case Form::ImadWide: {   // h_imad_wide
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      decl_src64(w, o.s[2], 2);
      const unsigned cmod = o.s[2].mod;
      w.f("    u32 rlo[32], rhi[32], carry;\n");
      if (cmod) {
        w.f("    u32 mlo[32], mhi[32];\n    for (unsigned l = 0; l < 32; ++l) {\n      u64 cc = L2[l] | (static_cast<u64>(H2[l]) << 32);\n");
        if (cmod & 1u) w.f("      cc = 0 - cc;\n");
        if (cmod & 2u) w.f("      cc = ~cc;\n");
        w.f("      mlo[l] = static_cast<u32>(cc); mhi[l] = static_cast<u32>(cc >> 32);\n    }\n    L2 = mlo; H2 = mhi;\n");
      }
      w.f("    imad_wide_k(rlo, rhi, A0, A1, L2, H2, %s, %s, &carry);\n", (o.aux & 2) ? "true" : "false", (o.aux & 1) ? pm(w, o.p, pa, sizeof pa) : "0u");
      if (o.dst != kNoReg) {
        store(w, o.dst, "rlo");
        if (o.dst + 1u != kRZ) store(w, static_cast<uint16_t>(o.dst + 1), "rhi");
      }
      if (o.dst2 != kNoReg) setp(w, o.dst2, "carry");
      break;
    }
    case Form::ImadHi: {   // h_imad_hi
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      const bool sgn = o.aux & 2, pair = o.aux & 4;
      if (pair) decl_src64(w, o.s[2], 2);
      else decl_src(w, o.s[2], 2, false);
      const bool neg = o.s[2].mod & 1u;
      w.f("    const u32 cin = %s;\n    u32 r[32], carry = 0;\n", (o.aux & 1) ? pm(w, o.p, pa, sizeof pa) : "0u");
      w.f("    for (unsigned l = 0; l < 32; ++l) {\n");
      if (sgn) w.f("      const u64 prod = static_cast<u64>(static_cast<int64_t>(static_cast<int32_t>(A0[l])) * static_cast<int32_t>(A1[l]));\n");
      else w.f("      const u64 prod = static_cast<u64>(A0[l]) * A1[l];\n");
      w.f("      u64 cc;\n");
      if (pair) {
        w.f("      cc = L2[l] | (static_cast<u64>(H2[l]) << 32);\n");
        if (neg) w.f("      cc = 0 - cc;\n");
      } else {
        w.f("      cc = static_cast<u64>(%s);\n", sgn ? "static_cast<int64_t>(static_cast<int32_t>(A2[l]))" : "static_cast<u64>(A2[l])");
      }
      w.f("      const unsigned __int128 sum = static_cast<unsigned __int128>(prod) + cc + ((cin >> l) & 1u);\n"
          "      r[l] = static_cast<u32>(sum >> 32);\n"
          "      carry |= static_cast<u32>((sum >> 64) != 0) << l;\n    }\n");
      store(w, o.dst, "r");
      if (o.dst2 != kNoReg) setp(w, o.dst2, "carry");
      break;
    }
    case Form::Lea: {   // h_lea
      const bool hi = o.aux & 1, x = o.aux & 2, three = o.aux & 4, bneg = o.aux & 8;
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      if (three) decl_src(w, o.s[2], 2, false);
      w.f("    const unsigned sh = %uu;\n    const u32 cin = %s;\n    u32 r[32], carry = 0;\n", o.a, x ? pm(w, o.p, pa, sizeof pa) : "0u");
      if (!hi && !x && !bneg) {
        w.f("    for (unsigned l = 0; l < 32; ++l) r[l] = (sh >= 32 ? 0u : A0[l] << sh) + A1[l];\n");
        if (o.dst2 != kNoReg) w.f("    for (unsigned l = 0; l < 32; ++l) carry |= static_cast<u32>(r[l] < (sh >= 32 ? 0u : A0[l] << sh)) << l;\n");
      } else {
        w.f("    for (unsigned l = 0; l < 32; ++l) {\n      u32 shifted;\n");
        if (!hi) w.f("      shifted = sh >= 32 ? 0u : A0[l] << sh;\n");
        else {
          if (three) w.f("      const u32 tp = A2[l];\n");
          else w.f("      const u32 tp = static_cast<u32>(static_cast<int32_t>(A0[l]) >> 31);\n");
          w.f("      shifted = static_cast<u32>(((A0[l] | (static_cast<u64>(tp) << 32)) << sh) >> 32);\n");
        }
        w.f("      u64 sum = static_cast<u64>(shifted) + A1[l] + ((cin >> l) & 1u);\n");
        if (bneg) w.f("      if (A1[l] == 0) sum += u64{1} << 32;\n");
        w.f("      r[l] = static_cast<u32>(sum);\n      carry |= static_cast<u32>((sum >> 32) != 0) << l;\n    }\n");
      }
      store(w, o.dst, "r");
      if (o.dst2 != kNoReg) setp(w, o.dst2, "carry");
      break;
    }
    case Form::Shf:   // h_shf
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32]; shf_k(r, A0, A1, A2, %uu, %s, %s, %s);\n", o.a, (o.aux & 1) ? "true" : "false", (o.aux & 2) ? "true" : "false", (o.aux & 4) ? "true" : "false");
      store(w, o.dst, "r");
      break;
    case Form::Isetp: {   // h_isetp
      static const unsigned kIntCmp[] = {0, 2, 0, 3, 4, 1, 5};
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      w.f("    u32 r = static_cast<u32>(icmp_mask_k(%uu, %s, A0, A1, 32)) & ex;\n    const u32 acc = %s;\n", kIntCmp[o.a], o.c ? "true" : "false", pm(w, o.p, pa, sizeof pa));
      if (mut(g_form_name)) w.f("    r ^= 4u;\n");
      setp2(w, o, true);
      break;
    }
    case Form::Fsetp:   // h_fsetp (the plain form: no |x| or -x)
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      w.f("    u32 r = fcmp_mask_k(%uu, %s, A0, A1) & ex;\n    const u32 acc = %s;\n", o.a, o.c ? "true" : "false", pm(w, o.p, pa, sizeof pa));
      if (mut(g_form_name)) w.f("    r ^= 4u;\n");
      setp2(w, o, true);
      break;
    case Form::Fsel:   // h_fsel
      decl_src(w, o.s[0], 0, true);
      decl_src(w, o.s[1], 1, true);
      w.f("    u32 r[32]; cprow(r, A1);\n    blend_m_k(r, A0, %s);\n", pm(w, o.p, pa, sizeof pa));
      store(w, o.dst, "r");
      break;
    case Form::Ffma:   // h_ffma (no directed rounding: that stays an escape)
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, true);
      if (!o.a) w.f("    u32 r[32]; ffma_k(r, A0, A1, A2);\n");
      else w.f("    u32 r[32]; ffma_x_k(r, A0, A1, A2, %uu);\n", o.a);
      store(w, o.dst, "r");
      break;
    case Form::Fadd:   // h_fadd
      for (int k = 0; k < 2; ++k) decl_src(w, o.s[k], k, true);
      if (!o.a) w.f("    u32 r[32]; fadd_k(r, A0, A1);\n");
      else w.f("    u32 r[32]; fadd_x_k(r, A0, A1, %uu);\n", o.a);
      store(w, o.dst, "r");
      break;
    case Form::Fmul:   // h_fmul
      for (int k = 0; k < 2; ++k) decl_src(w, o.s[k], k, true);
      if (!o.a) w.f("    u32 r[32]; fmul_k(r, A0, A1);\n");
      else w.f("    u32 r[32]; fmul_x_k(r, A0, A1, %uu);\n", o.a);
      store(w, o.dst, "r");
      break;
    case Form::Prmt:   // h_prmt, default mode (a = 0)
      for (int k = 0; k < 3; ++k) decl_src(w, o.s[k], k, false);
      w.f("    u32 r[32]; prmt0_k(r, A0, A2, A1);\n");
      store(w, o.dst, "r");
      break;
    case Form::Imnmx:   // h_imnmx: the minimum on the lanes whose predicate holds, the maximum on the others
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      w.f("    const u32 p = %s;\n    u32 r[32];\n    for (unsigned l = 0; l < 32; ++l) {\n", pm(w, o.p, pa, sizeof pa));
      if (o.c) w.f("      const int32_t x = static_cast<int32_t>(A0[l]), y = static_cast<int32_t>(A1[l]);\n      const int32_t mn = x < y ? x : y, mx = x > y ? x : y;\n"
                   "      r[l] = static_cast<u32>((p >> l) & 1u ? mn : mx);\n");
      else w.f("      const u32 x = A0[l], y = A1[l];\n      const u32 mn = x < y ? x : y, mx = x > y ? x : y;\n      r[l] = (p >> l) & 1u ? mn : mx;\n");
      w.f("    }\n");
      store(w, o.dst, "r");
      break;
    case Form::I2fp:   // h_i2fp
      decl_src(w, o.s[0], 0, false);
      w.f("    u32 r[32]; i2f_k(%s, r, A0);\n", o.c ? "true" : "false");
      store(w, o.dst, "r");
      break;
    case Form::Sel:   // h_sel: b where the predicate is clear, a where it holds
      decl_src(w, o.s[0], 0, false);
      decl_src(w, o.s[1], 1, false);
      w.f("    u32 r[32]; cprow(r, A1);\n    blend_m_k(r, A0, %s);\n", pm(w, o.p, pa, sizeof pa));
      store(w, o.dst, "r");
      break;
    case Form::Plop3: {   // h_plop3: for each lane the table bit the three predicates select, into two predicates
      char pc[96];
      w.f("    const u32 pa = %s, pb = %s, pd = %s;\n    u32 r0 = 0, r1 = 0;\n", pm(w, o.s[0], pa, sizeof pa), pm(w, o.s[1], pb, sizeof pb), pm(w, o.s[2], pc, sizeof pc));
      for (unsigned idx = 0; idx < 8; ++idx) {
        char term[160];
        std::snprintf(term, sizeof term, "%spa & %spb & %spd", (idx & 4) ? "" : "~", (idx & 2) ? "" : "~", (idx & 1) ? "" : "~");
        if ((o.a >> idx) & 1) w.f("    r0 |= %s;\n", term);
        if ((o.b >> idx) & 1) w.f("    r1 |= %s;\n", term);
      }
      if (mut(g_form_name)) w.f("    r0 ^= 8u;\n");
      setp(w, o.dst, "r0");
      setp(w, o.dst2, "r1");
      break;
    }
    case Form::S2r:   // h_s2r: the host fills the executing lanes (it needs the block and the warp)
      if (o.dst == kNoReg) break;
      w.f("    u32 r[32] = {};\n    c->sreg(c, %uu, r, ex);\n", o.a);
      store(w, o.dst, "r");   // (the form's mutant is in store())
      break;
    case Form::Cs2r:   // h_cs2r, SRZ: zero (and the register after it for a pair)
      if (o.dst == kNoReg) break;
      w.f("    u32 r[32]; splat(r, %uu);\n", mut(g_form_name) ? 1u : 0u);
      store(w, o.dst, "r");
      if (o.b && o.dst + 1u != kRZ) store(w, static_cast<uint16_t>(o.dst + 1), "r");
      break;
    case Form::Umov:   // h_umov: the first executing lane's value into a uniform register
      w.f("    const unsigned l = static_cast<unsigned>(__builtin_ctz(ex));\n");
      if (o.dst != kNoDst) w.f("    c->ur[%u] = %s%s;\n", unsigned{o.dst}, usrc(o.s[0]).c_str(), mut(g_form_name) ? " ^ 1u" : "");
      break;
    case Form::Uldc: {   // h_uldc, bank 0: a word, a narrow field, or two words
      const unsigned size = o.b;
      const uint64_t off = static_cast<uint64_t>(o.imm);
      if (size == 5) {
        if (o.dst != kNoDst) {
          w.f("    c->ur[%u] = b0w(c, %lluu)%s;\n", unsigned{o.dst}, static_cast<unsigned long long>(off), mut(g_form_name) ? " ^ 1u" : "");
          if (o.dst + 1u < o.c) w.f("    c->ur[%u] = b0w(c, %lluu);\n", o.dst + 1u, static_cast<unsigned long long>(off + 4));
        }
        break;
      }
      if (o.dst == kNoDst) break;
      char v[200];
      if (size < 4) std::snprintf(v, sizeof v, "b0w(c, %lluu) >> %uu", static_cast<unsigned long long>(off & ~uint64_t{3}), static_cast<unsigned>(8 * (off & 3)));
      else std::snprintf(v, sizeof v, "b0w(c, %lluu)", static_cast<unsigned long long>(off));
      const char* ext = size == 0 ? "(%s) & 0xffu" : size == 1 ? "static_cast<u32>(static_cast<int8_t>(%s))" : size == 2 ? "(%s) & 0xffffu" : size == 3 ? "static_cast<u32>(static_cast<int16_t>(%s))" : "%s";
      char e[300];
      std::snprintf(e, sizeof e, ext, v);
      w.f("    c->ur[%u] = %s%s;\n", unsigned{o.dst}, e, mut(g_form_name) ? " ^ 1u" : "");
      break;
    }
    case Form::Uiadd3: {   // h_uiadd3
      w.f("    const unsigned l = static_cast<unsigned>(__builtin_ctz(ex));\n    u64 sum = 0;\n");
      for (int k = 0; k < 3; ++k) {
        if ((o.b >> (2 * k)) & 1) w.f("    sum += static_cast<u32>(~%s); sum += 1;\n", usrc(o.s[k]).c_str());
        else if ((o.b >> (2 * k + 1)) & 1) w.f("    sum += static_cast<u32>(~%s);\n", usrc(o.s[k]).c_str());
        else w.f("    sum += %s;\n", usrc(o.s[k]).c_str());
      }
      if (o.aux & 1) w.f("    sum += static_cast<u64>((%s >> l) & 1u) + static_cast<u64>((%s >> l) & 1u);\n", pm(w, o.p, pa, sizeof pa), pm(w, o.p2, pb, sizeof pb));
      if (o.dst != kNoDst) w.f("    c->ur[%u] = static_cast<u32>(sum)%s;\n", unsigned{o.dst}, mut(g_form_name) ? " ^ 1u" : "");
      w.f("    const u64 over = sum >> 32;\n");
      if (o.dst2 != kNoDst && o.dst2 != 7) w.f("    c->up[%u] = over >= 1;\n", unsigned{o.dst2});
      if (o.a != kNoDst && o.a != 7) w.f("    c->up[%u] = over >= 2;\n", o.a);
      break;
    }
    case Form::Ulop3: {   // h_ulop3
      w.f("    const unsigned l = static_cast<unsigned>(__builtin_ctz(ex));\n    const u32 ua = %s, ub = %s, uc = %s;\n    u32 r = 0;\n", usrc(o.s[0]).c_str(), usrc(o.s[1]).c_str(),
          usrc(o.s[2]).c_str());
      for (unsigned idx = 0; idx < 8; ++idx)
        if ((o.a >> idx) & 1) w.f("    r |= %sua & %sub & %suc;\n", (idx & 4) ? "" : "~", (idx & 2) ? "" : "~", (idx & 1) ? "" : "~");
      if (o.dst != kNoDst) w.f("    c->ur[%u] = r%s;\n", unsigned{o.dst}, mut(g_form_name) ? " ^ 1u" : "");
      break;
    }
    case Form::Usel:   // h_usel
      w.f("    const unsigned l = static_cast<unsigned>(__builtin_ctz(ex));\n");
      if (o.dst != kNoDst)
        w.f("    c->ur[%u] = (((%s >> l) & 1u) ? %s : %s)%s;\n", unsigned{o.dst}, pm(w, o.p, pa, sizeof pa), usrc(o.s[0]).c_str(), usrc(o.s[1]).c_str(),
            mut(g_form_name) ? " ^ 1u" : "");
      break;
    case Form::Uisetp: {   // h_uisetp: two uniform predicates
      w.f("    const unsigned l = static_cast<unsigned>(__builtin_ctz(ex));\n    const u32 ua = %s, ub = %s;\n    const bool acc = (%s >> l) & 1u;\n", usrc(o.s[0]).c_str(),
          usrc(o.s[1]).c_str(), pm(w, o.p, pa, sizeof pa));
      w.f("    const bool r = icmp1(%uu, %s, ua, ub);\n", o.a, o.c ? "true" : "false");
      if (o.dst != kNoDst && o.dst != 7) w.f("    c->up[%u] = bool_op1(%uu, r, acc)%s;\n", unsigned{o.dst}, o.b, mut(g_form_name) ? " == false" : "");
      if (o.dst2 != kNoDst && o.dst2 != 7) w.f("    c->up[%u] = bool_op1(%uu, !r, acc);\n", unsigned{o.dst2}, o.b);
      break;
    }
    default: break;
  }
  (void)flt;
  w.f("    }\n");
  return w.s;
}

}  // namespace vgpu::sass::jit
