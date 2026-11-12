#!/usr/bin/env bash
# scripts/apt-mirror-fallback.sh rewrites apt's mirror list only when its first host does not resolve, and every
# hosted-runner "apt-get update" in the workflows and action.yml runs it first.
set -u
here=$(cd "$(dirname "$0")/../.." && pwd)
script=$here/scripts/apt-mirror-fallback.sh
fail=0
check() { if [[ $2 == ok ]]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
mk_etc() {
  rm -rf "$tmp/etc"; mkdir -p "$tmp/etc/sources.list.d"
  printf 'http://azure.archive.ubuntu.com/ubuntu/\tpriority:1\nhttp://archive.ubuntu.com/ubuntu/\tpriority:2\n' > "$tmp/etc/apt-mirrors.txt"
  printf 'Types: deb\nURIs: mirror+file:%s/etc/apt-mirrors.txt\nSuites: noble noble-updates\n\nTypes: deb\nURIs: mirror+file:%s/etc/apt-mirrors.txt\nSuites: noble-security\n' "$tmp" "$tmp" > "$tmp/etc/sources.list.d/ubuntu.sources"
  printf 'deb mirror+file:%s/etc/apt-mirrors.txt jammy main\n' "$tmp" > "$tmp/etc/sources.list"
}
fake_getent() {  # the hosts the fake resolver answers for (all others fail)
  printf '#!/bin/sh\necho "$2" >> "%s/asked"\nfor h in %s; do [ "$2" = "$h" ] && exit 0; done\nexit 2\n' "$tmp" "$*" > "$tmp/bin/getent"
  chmod +x "$tmp/bin/getent"
}
run() { PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" 2>&1; }
uris() { cat "$tmp/etc/sources.list.d/ubuntu.sources" "$tmp/etc/sources.list"; }

# 1. Only the image's mirror host fails: archive.ubuntu.com is used.
mk_etc; rm -f "$tmp/asked"; fake_getent archive.ubuntu.com
out=$(run); rc=$?
check "an unresolvable mirror exits 0" "$([[ $rc == 0 ]] && echo ok)"
check "it asks about the first mirror host" "$([[ $(head -n1 "$tmp/asked") == azure.archive.ubuntu.com ]] && echo ok)"
check "it says which mirror it chose" "$([[ $out == *"apt uses archive.ubuntu.com instead"* ]] && echo ok)"
check "deb822 sources point at archive.ubuntu.com, security included" \
  "$([[ $(grep -c 'URIs: http://archive.ubuntu.com/ubuntu/' "$tmp/etc/sources.list.d/ubuntu.sources") == 2 ]] && echo ok)"
check "sources.list points at archive.ubuntu.com" "$(grep -q 'deb http://archive.ubuntu.com/ubuntu/ jammy' "$tmp/etc/sources.list" && echo ok)"
check "no mirror+file entry is left" "$(! grep -rq 'mirror+file:' "$tmp/etc" && echo ok)"
run >/dev/null
check "a second run is harmless" "$([[ $? == 0 ]] && echo ok)"

# 2. The whole ubuntu.com domain fails to resolve (the 2026-11-11 outage): a mirror outside it is used.
mk_etc; fake_getent mirrors.edge.kernel.org
out=$(run)
check "with ubuntu.com unresolvable, kernel.org's mirror is chosen" "$([[ $out == *"apt uses mirrors.edge.kernel.org instead"* ]] && echo ok)"
check "sources carry its path" "$([[ $(uris | grep -c 'http://mirrors.edge.kernel.org/ubuntu/') == 3 ]] && echo ok)"
mk_etc; fake_getent mirror.us.leaseweb.net
out=$(run)
check "the last candidate is reachable too" "$([[ $out == *"apt uses mirror.us.leaseweb.net instead"* ]] && echo ok)"

# 3. Nothing resolves: apt is left alone, and the log says so.
mk_etc; before=$(uris); fake_getent
out=$(run); rc=$?
check "when no mirror resolves it exits 0 and changes nothing" "$([[ $rc == 0 && $(uris) == "$before" ]] && echo ok)"
check "and says so" "$([[ $out == *"neither does any fallback mirror"* ]] && echo ok)"

# 4. The image's mirror resolves: nothing is touched, nothing is printed.
mk_etc; before=$(uris); fake_getent azure.archive.ubuntu.com archive.ubuntu.com
out=$(run)
check "a resolvable mirror changes nothing" "$([[ $(uris) == "$before" && -z $out ]] && echo ok)"

# 5. A container image has no mirror list.
rm -rf "$tmp/etc"; mkdir -p "$tmp/etc"; fake_getent
run >/dev/null
check "no mirror list (a container image) is a no-op" "$([[ $? == 0 ]] && echo ok)"

# Every hosted-runner apt-get update runs it first (the self-hosted pool steps take a flock and are not hosted).
missing=$(cd "$here" && python3 -I - <<'PY'
import glob, re
bad = []
for f in glob.glob(".github/workflows/*.yml") + ["action.yml"]:
    lines = open(f).read().split("\n")
    for i, l in enumerate(lines):
        if re.match(r"^\s*(sudo|\$SUDO) apt-get update -qq\s*$", l) and "apt-mirror-fallback.sh" not in lines[i - 1]:
            bad.append(f"{f}:{i + 1}")
print("\n".join(bad))
PY
)
check "every hosted apt-get update runs the fallback first${missing:+ (missing: $(echo $missing | tr '\n' ' '))}" "$([[ -z $missing ]] && echo ok)"

exit $fail
