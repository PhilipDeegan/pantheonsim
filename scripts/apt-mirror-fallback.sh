#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners point apt at a mirror list (mirror+file:/etc/apt/apt-mirrors.txt) whose first host is
# Azure's azure.archive.ubuntu.com. When that name does not resolve, every apt-get update/install in a job fails
# ("Temporary failure resolving 'azure.archive.ubuntu.com'", exit 100), whatever the job does. In that case, and only
# then, point apt at the first of a few full Ubuntu mirrors whose name does resolve (archive.ubuntu.com first, then
# mirrors outside ubuntu.com, in case the runner's resolver is failing for that whole domain, as it did on 2026-11-11);
# each serves every pocket, security included. On a healthy runner, in a container image with no mirror list, or on a
# machine whose mirror resolves, this changes nothing; if none of them resolves either, it changes nothing and says so.
#
#   APT_ETC     the apt configuration directory (default /etc/apt; the test points it at a temporary one)
set -u
etc=${APT_ETC:-/etc/apt}
list=$etc/apt-mirrors.txt
[[ -r $list ]] || exit 0
host=$(head -n 1 "$list" | sed -E 's#^[a-z+]*://##; s#[/[:space:]].*$##')
[[ -n $host ]] || exit 0
getent hosts "$host" >/dev/null 2>&1 && exit 0

# host/path of each candidate, in order of preference
candidates=(archive.ubuntu.com/ubuntu mirrors.edge.kernel.org/ubuntu mirror.math.princeton.edu/pub/ubuntu
            mirrors.xtom.com/ubuntu mirror.us.leaseweb.net/ubuntu)
chosen=
for c in "${candidates[@]}"; do
  if getent hosts "${c%%/*}" >/dev/null 2>&1; then chosen=$c; break; fi
done
if [[ -z $chosen ]]; then
  echo "::warning::$host does not resolve, and neither does any fallback mirror (${candidates[*]%%/*}); apt is left as it is"
  exit 0
fi
echo "::warning::$host does not resolve; apt uses ${chosen%%/*} instead"
for f in "$etc/sources.list" "$etc"/sources.list.d/*.sources "$etc"/sources.list.d/*.list; do
  [[ -f $f ]] && grep -q 'mirror+file:' "$f" || continue
  sudo=; [[ -w $f ]] || sudo=sudo
  $sudo sed -i -E "s#mirror\\+file:$list#http://$chosen/#g" "$f"
done
exit 0
