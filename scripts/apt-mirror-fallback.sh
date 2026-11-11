#!/usr/bin/env bash
# GitHub-hosted Ubuntu runners point apt at a mirror list (mirror+file:/etc/apt/apt-mirrors.txt) whose first host is
# Azure's azure.archive.ubuntu.com. When that name does not resolve, every apt-get update/install in a job fails
# ("Temporary failure resolving 'azure.archive.ubuntu.com'", exit 100), whatever the job does. In that case, and only
# then, point apt at archive.ubuntu.com instead (it serves every pocket, security included). On a healthy runner, in a
# container image with no mirror list, or on a machine whose mirror resolves, this changes nothing.
#
#   APT_ETC     the apt configuration directory (default /etc/apt; the test points it at a temporary one)
set -u
etc=${APT_ETC:-/etc/apt}
list=$etc/apt-mirrors.txt
[[ -r $list ]] || exit 0
host=$(head -n 1 "$list" | sed -E 's#^[a-z+]*://##; s#[/[:space:]].*$##')
[[ -n $host ]] || exit 0
getent hosts "$host" >/dev/null 2>&1 && exit 0

echo "::warning::$host does not resolve; apt uses archive.ubuntu.com instead"
for f in "$etc/sources.list" "$etc"/sources.list.d/*.sources "$etc"/sources.list.d/*.list; do
  [[ -f $f ]] && grep -q 'mirror+file:' "$f" || continue
  sudo=; [[ -w $f ]] || sudo=sudo
  # shellcheck disable=SC2016  # the sed program is single-quoted on purpose
  $sudo sed -i -E "s#mirror\\+file:$list#http://archive.ubuntu.com/ubuntu/#g" "$f"
done
exit 0
