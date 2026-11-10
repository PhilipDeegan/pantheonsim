// Exact parallel launches: the shadow of device memory a trial of one launch is run under, and its undo log
// (vgpu/memory.hpp, "exact parallel launches"; docs/exact-threads.md has the argument for why a trial that commits
// left the bits of the serial run).
//
// One 32-bit word stands for four bytes of the device's address range [va_base_, high_water_va_):
//     0                 no block has touched the bytes
//     b + 1             block b has read them, and no other block has touched them
//     kMany             more than one block has read them, and none has written them
//     kWrite | (b + 1)  block b has written them (and may have read them); no other block has touched them
// A word only moves up that list (0 -> one reader -> many readers; 0 or its own reader mark -> its writer), by
// compare-and-swap, and a block makes its access only after the word allows it. So for any two blocks and any byte:
//   * both read it: nothing to tell apart;
//   * one writes it and the other reads or writes it: whichever came second to the word found a state it may not
//     enter, and stopped the trial BEFORE touching the byte.
// A stopped trial is undone from the bytes each block kept when it first wrote a word (it owned the word then, so
// what it kept is what the launch started with), and the launch is run again in order.
#include "vgpu/memory.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <sys/mman.h>

namespace vgpu {

class MemoryManager::Spec {
 public:
  using Word = std::atomic<uint32_t>;
  static constexpr uint32_t kMany = 0xFFFFFFFFu, kWrite = 0x80000000u;
  static constexpr uint64_t kGranule = 4;          // bytes a word stands for
  static constexpr uint64_t kPageWords = 1024;     // words of a shadow page: 4 KiB of device memory, 4 KiB of shadow
  static constexpr size_t kPoolPages = 65536;      // shadow pages kept between trials (256 MiB of host memory at most)
  static constexpr uint64_t kKeepLimit = uint64_t{1} << 30;   // bytes one worker may keep for undoing before the trial stops

  struct Worker {
    uint32_t tag = 0;                    // block + 1 of the block it runs now
    std::vector<Word*> fresh;            // zeroed pages ready to install
    std::vector<uint64_t> installed;     // the top-table slots whose pages this worker installed
    std::vector<std::pair<uint64_t, uint64_t>> kept_at;   // (address, length) of each kept piece, in the order kept
    std::vector<uint8_t> kept;                            // their bytes, one after another
    // The last page asked for (a block's accesses are mostly near one another).
    uint64_t last_slot = ~uint64_t{0};
    Word* last_page = nullptr;
  };

  ~Spec() {
    for (Word* p : pool) std::free(p);
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Word*>));
  }

  // The top table: a slot per shadow page of the range, zero pages of the kernel's until a slot is written.
  void size_for(uint64_t base_va, uint64_t bytes) {
    base = base_va;
    span = bytes;
    const uint64_t slots = (bytes / kGranule + kPageWords - 1) / kPageWords + 1;
    if (slots <= top_slots) return;
    if (top) ::munmap(top, top_slots * sizeof(std::atomic<Word*>));
    top_slots = std::max<uint64_t>(slots * 2, 4096);
    void* m = ::mmap(nullptr, top_slots * sizeof(std::atomic<Word*>), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) {
      top = nullptr;
      top_slots = 0;
      throw std::bad_alloc();
    }
    top = static_cast<std::atomic<Word*>*>(m);   // (zero bytes: null pointers)
  }

  Word* fresh_page(Worker& w) {
    if (w.fresh.empty()) {
      std::lock_guard<std::mutex> g(pool_mu);
      for (int i = 0; i < 16 && !pool.empty(); ++i) {
        w.fresh.push_back(pool.back());
        pool.pop_back();
      }
    }
    if (!w.fresh.empty()) {
      Word* p = w.fresh.back();
      w.fresh.pop_back();
      return p;
    }
    void* m = std::calloc(kPageWords, sizeof(Word));
    if (!m) throw std::bad_alloc();
    return static_cast<Word*>(m);
  }

  // The word of granule `g` (below span / kGranule).
  [[gnu::always_inline]] Word& word(Worker& w, uint64_t g) {
    const uint64_t slot = g / kPageWords;
    if (slot != w.last_slot) [[unlikely]] {
      Word* page = top[slot].load(std::memory_order_acquire);
      if (!page) {
        Word* mine = fresh_page(w);
        if (top[slot].compare_exchange_strong(page, mine, std::memory_order_acq_rel)) {
          page = mine;
          w.installed.push_back(slot);
        } else {
          w.fresh.push_back(mine);   // another worker's page is there
        }
      }
      w.last_slot = slot;
      w.last_page = page;
    }
    return w.last_page[g % kPageWords];
  }

  std::atomic<Word*>* top = nullptr;
  uint64_t top_slots = 0;
  uint64_t base = 0, span = 0;
  bool host_maps = false;
  std::atomic<bool> stopped{false};
  std::atomic<const char*> why{nullptr};
  std::vector<Worker> workers;
  std::mutex pool_mu;
  std::vector<Word*> pool;   // zeroed pages
};

namespace {
thread_local MemoryManager::Spec::Worker* t_spec_worker = nullptr;
}  // namespace

bool MemoryManager::spec_possible() const {
  if (memo_sink_ || memo_untrackable() || memo_faults_armed() || memo_trace_active()) return false;
  if (card_gate_.load(std::memory_order_relaxed) || card_resident_.load(std::memory_order_relaxed) > 0) return false;
  SharedGuard table_guard(table_lock_.get());
  for (const auto& [base, a] : live_)
    if (a.external) return false;   // bytes that live in a file another process may map
  return high_water_va_ > va_base_;
}

void MemoryManager::spec_begin(unsigned workers) {
  if (!spec_kept_) spec_kept_ = std::make_shared<Spec>();
  Spec& s = *spec_kept_;
  s.size_for(va_base_, high_water_va_ - va_base_);
  s.host_maps = memo_host_maps();
  s.stopped.store(false, std::memory_order_relaxed);
  s.why.store(nullptr, std::memory_order_relaxed);
  s.workers.clear();
  s.workers.resize(workers);
  spec_ = &s;
}

void MemoryManager::spec_enter(unsigned worker) { t_spec_worker = &spec_->workers[worker]; }

void MemoryManager::spec_block(uint64_t block) { t_spec_worker->tag = static_cast<uint32_t>(block + 1); }

void MemoryManager::spec_leave() { t_spec_worker = nullptr; }

bool MemoryManager::spec_stopped() const { return spec_ && spec_->stopped.load(std::memory_order_relaxed); }

void MemoryManager::spec_stop(const char* why) {
  if (!spec_) return;
  const char* none = nullptr;
  spec_->why.compare_exchange_strong(none, why, std::memory_order_relaxed);
  spec_->stopped.store(true, std::memory_order_release);
}

void MemoryManager::spec_refuse(const char* why) const {
  const char* none = nullptr;
  spec_->why.compare_exchange_strong(none, why, std::memory_order_relaxed);
  spec_->stopped.store(true, std::memory_order_release);
  throw SpecStop{};
}

MemoryManager::SpecOutcome MemoryManager::spec_end() {
  Spec& s = *spec_;
  spec_ = nullptr;   // (the workers have left: what follows is this thread's alone, and write() below is no trial's)
  SpecOutcome out;
  out.committed = !s.stopped.load(std::memory_order_acquire);
  if (const char* why = s.why.load(std::memory_order_relaxed)) out.why = why;
  for (Spec::Worker& w : s.workers) {
    out.kept_bytes += w.kept.size();
    out.shadow_pages += w.installed.size();
    if (!out.committed) {
      // Each piece is what the launch started with at bytes only this worker's blocks wrote, so the order of putting back is free.
      uint64_t at = 0;
      for (const auto& [addr, len] : w.kept_at) {
        write(addr, w.kept.data() + at, len);
        at += len;
      }
    }
  }
  // The shadow goes back to nothing: every installed page is zeroed and pooled, its slot cleared.
  std::lock_guard<std::mutex> g(s.pool_mu);
  for (Spec::Worker& w : s.workers) {
    for (const uint64_t slot : w.installed) {
      Spec::Word* page = s.top[slot].exchange(nullptr, std::memory_order_acq_rel);
      if (s.pool.size() < Spec::kPoolPages) {
        std::memset(static_cast<void*>(page), 0, Spec::kPageWords * sizeof(Spec::Word));
        s.pool.push_back(page);
      } else {
        std::free(page);
      }
    }
    for (Spec::Word* page : w.fresh) {   // zeroed and never installed
      if (s.pool.size() < Spec::kPoolPages) s.pool.push_back(page);
      else std::free(page);
    }
  }
  s.workers.clear();
  return out;
}

void MemoryManager::spec_keep(uint64_t addr, uint64_t len) const {
  Spec::Worker& w = *t_spec_worker;
  if (w.kept.size() + len > Spec::kKeepLimit) spec_refuse("the undo log passed its limit");
  SharedGuard table_guard(table_lock_.get());
  const uint64_t end = addr + len;
  while (addr < end) {
    auto up = live_.upper_bound(addr);
    if (up == live_.begin() || addr >= std::prev(up)->first + std::prev(up)->second.size) {
      // No allocation holds this byte (padding after one, a freed range): nothing a store can change. On to the next allocation.
      if (up == live_.end() || up->first >= end) return;
      addr = up->first;
      continue;
    }
    const auto it = std::prev(up);
    const uint64_t n = std::min(end, it->first + it->second.size) - addr;
    const size_t at = w.kept.size();
    w.kept.resize(at + n);
    read_chunks(it->second, addr - it->first, w.kept.data() + at, n);
    w.kept_at.emplace_back(addr, n);
    addr += n;
  }
}

// A block reads [addr, addr + len).
void MemoryManager::spec_read(uint64_t addr, uint64_t len) const {
  Spec& s = *spec_;
  if (s.stopped.load(std::memory_order_relaxed)) throw SpecStop{};
  Spec::Worker* wp = t_spec_worker;
  if (!wp) spec_refuse("a thread outside the launch used device memory");
  if (len == 0) return;
  const uint64_t off = addr - s.base;
  // Outside the device's range: host memory (pinned, managed), which no block of a trial can write (spec_write), or
  // nothing at all, which the access itself reports.
  if (off >= s.span) return;
  Spec::Worker& w = *wp;
  const uint32_t tag = w.tag;
  const uint64_t g1 = (std::min(off + len, s.span) - 1) / Spec::kGranule;
  for (uint64_t g = off / Spec::kGranule; g <= g1; ++g) {
    Spec::Word& word = s.word(w, g);
    uint32_t v = word.load(std::memory_order_acquire);
    for (;;) {
      if (v == Spec::kMany || v == tag || v == (tag | Spec::kWrite)) break;
      if (v == 0) {
        if (word.compare_exchange_weak(v, tag, std::memory_order_acq_rel)) break;
      } else if (v & Spec::kWrite) {
        spec_refuse("a block read what another block wrote");
      } else if (word.compare_exchange_weak(v, Spec::kMany, std::memory_order_acq_rel)) {
        break;
      }
    }
  }
}

// A block is about to write [addr, addr + len): it comes to own every word of it, and the bytes of the words it did
// not own yet are kept first.
void MemoryManager::spec_write(uint64_t addr, uint64_t len) const {
  Spec& s = *spec_;
  if (s.stopped.load(std::memory_order_relaxed)) throw SpecStop{};
  Spec::Worker* wp = t_spec_worker;
  if (!wp) spec_refuse("a thread outside the launch used device memory");
  if (len == 0) return;
  const uint64_t off = addr - s.base;
  // A store the shadow has no word for (host memory another thread or process may share; an address that is no
  // allocation's) cannot be undone or checked: the launch runs in order.
  if (off >= s.span || len > s.span - off) spec_refuse("a block wrote outside the device's own memory");
  if (s.host_maps && host_maps_ && host_maps_->may_overlap(addr, addr + len - 1)) spec_refuse("a block wrote host-mapped memory");
  Spec::Worker& w = *wp;
  const uint32_t mine = w.tag | Spec::kWrite;
  const uint64_t g1 = (off + len - 1) / Spec::kGranule;
  uint64_t run = ~uint64_t{0};   // the first granule of the run of words taken just now, not kept yet
  const auto keep = [&](uint64_t upto) {
    if (run == ~uint64_t{0}) return;
    spec_keep(s.base + run * Spec::kGranule, (upto - run) * Spec::kGranule);
    run = ~uint64_t{0};
  };
  for (uint64_t g = off / Spec::kGranule; g <= g1; ++g) {
    Spec::Word& word = s.word(w, g);
    uint32_t v = word.load(std::memory_order_acquire);
    for (;;) {
      if (v == mine) {
        keep(g);
        break;
      }
      if (v == 0 || v == w.tag) {
        if (!word.compare_exchange_weak(v, mine, std::memory_order_acq_rel)) continue;
        if (run == ~uint64_t{0}) run = g;
        break;
      }
      // Another block's word, or one several blocks read. The words taken in this call are still unwritten: kept or not, undoing
      // leaves them as they are.
      spec_refuse(v != Spec::kMany && (v & Spec::kWrite) ? "two blocks wrote the same bytes" : "a block wrote what another block read");
    }
  }
  keep(g1 + 1);
}

void MemoryManager::spec_read_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  for (size_t i = 0; i < count; ++i) spec_read(addrs[i], size);
}

void MemoryManager::spec_write_each(const uint64_t* addrs, size_t count, uint32_t size) const {
  for (size_t i = 0; i < count; ++i) spec_write(addrs[i], size);
}

void MemoryManager::spec_write_strided(uint64_t a0, int64_t stride, size_t count, uint32_t size) const {
  for (size_t i = 0; i < count; ++i) spec_write(a0 + static_cast<uint64_t>(stride) * i, size);
}

}  // namespace vgpu
