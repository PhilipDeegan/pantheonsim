#!/usr/bin/env bash
# No build or test output, editor backup or merge leftover is tracked by git.
#
# Compiled objects are allowed only where tests read them on purpose (amd/tests/data,
# amd/tests/hipcc: the code objects a compiler produced, checked in so the tests need no
# compiler). *.log files are allowed only in the two places that keep run output as evidence
# or as a test oracle (docs/perf-design/data, nvidia/tests/data/card).
# Outside a git checkout (a source tarball) there is nothing to check.
set -euo pipefail
cd "$(dirname "$0")/../.."
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "not a git checkout: nothing to check"
  exit 0
fi
bad=$(git ls-files | grep -E '(^|/)(Testing|__pycache__|CMakeFiles)/|\.(pyc|orig|rej|bak|swp|tmp)$|~$|(^|/)(CMakeCache\.txt|core|nohup\.out)$' || true)
objs=$(git ls-files | grep -E '\.(o|a|so)$' | grep -Ev '^amd/tests/(data|hipcc)/' || true)
logs=$(git ls-files | grep -E '\.log$' | grep -Ev '^(docs/perf-design/data|nvidia/tests/data/card)/' || true)
all=$(printf '%s\n%s\n%s\n' "$bad" "$objs" "$logs" | sed '/^$/d')
if [[ -n $all ]]; then
  echo "tracked build outputs or leftovers (git rm them; .gitignore keeps new ones out):"
  echo "$all"
  exit 1
fi
echo "no tracked build outputs or leftovers"
