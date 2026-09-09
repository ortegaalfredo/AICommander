#!/usr/bin/env bash
#
# build_pypi.sh - build the aic-agent sdist and wheel into ./dist
#
# Usage:
#   ./build_pypi.sh               # build sdist + wheel
#   ./build_pypi.sh --skip-tests  # skip the offline unit-test suite
#
# Upload separately with ./upload_pypi.sh

set -euo pipefail
cd "$(dirname "$0")"

SKIP_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-tests) SKIP_TESTS=1 ;;
        -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Prefer the repo venv, fall back to PATH.
if [ -x "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

VERSION=$("$PYTHON" -c "import aic; print(aic.__version__)")
echo "[*] Building aic-agent $VERSION"

if ! "$PYTHON" -c "import build" >/dev/null 2>&1; then
    echo "[ERROR] 'build' is not installed. Run: $PYTHON -m pip install build" >&2
    exit 1
fi

if [ "$SKIP_TESTS" -ne 1 ]; then
    echo "[*] Running offline unit tests..."
    "$PYTHON" tests/test_truncation.py --skip-live | tail -1
fi

echo "[*] Cleaning old artifacts..."
rm -rf build dist *.egg-info
mkdir -p dist

echo "[*] Building sdist and wheel..."
"$PYTHON" -m build --outdir dist

echo "[*] Checking distributions..."
"$PYTHON" -m twine check dist/*

echo "[*] Done:"
ls -la dist/
