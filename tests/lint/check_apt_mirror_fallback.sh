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
fake_getent() {  # $1 = exit status of the fake resolver
  printf '#!/bin/sh\necho "$2" >> "%s/resolved"\nexit %s\n' "$tmp" "$1" > "$tmp/bin/getent"; chmod +x "$tmp/bin/getent"
}

mk_etc; fake_getent 2
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" 2>&1); rc=$?
check "an unresolvable mirror exits 0" "$([[ $rc == 0 ]] && echo ok)"
check "it asks about the first mirror host" "$([[ $(head -n1 "$tmp/resolved") == azure.archive.ubuntu.com ]] && echo ok)"
check "it warns" "$([[ $out == *"does not resolve"* ]] && echo ok)"
check "deb822 sources point at archive.ubuntu.com, security included" \
  "$([[ $(grep -c 'URIs: http://archive.ubuntu.com/ubuntu/' "$tmp/etc/sources.list.d/ubuntu.sources") == 2 ]] && echo ok)"
check "sources.list points at archive.ubuntu.com" "$(grep -q 'deb http://archive.ubuntu.com/ubuntu/ jammy' "$tmp/etc/sources.list" && echo ok)"
check "no mirror+file entry is left" "$(! grep -rq 'mirror+file:' "$tmp/etc" && echo ok)"
PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" >/dev/null 2>&1
check "a second run is harmless" "$([[ $? == 0 ]] && echo ok)"

mk_etc; fake_getent 0; before=$(cat "$tmp/etc/sources.list.d/ubuntu.sources" "$tmp/etc/sources.list")
out=$(PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" 2>&1)
check "a resolvable mirror changes nothing" "$([[ $(cat "$tmp/etc/sources.list.d/ubuntu.sources" "$tmp/etc/sources.list") == "$before" && -z $out ]] && echo ok)"

rm -rf "$tmp/etc"; mkdir -p "$tmp/etc"; fake_getent 2
PATH="$tmp/bin:$PATH" APT_ETC="$tmp/etc" "$script" >/dev/null 2>&1
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
