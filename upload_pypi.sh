#!/usr/bin/env bash
#
# upload_pypi.sh - publish the distributions in ./dist to PyPI.
#
# Usage:
#   ./upload_pypi.sh           # upload ./dist/* to pypi.org
#   ./upload_pypi.sh --test    # upload to test.pypi.org instead
#
# Credentials come from ~/.pypirc or the TWINE_API_TOKEN environment variable.
# Run ./build_pypi.sh first.

set -euo pipefail
cd "$(dirname "$0")"

REPO_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --test) REPO_ARGS=(--repository testpypi) ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[ERROR] unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ -x "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if ! "$PYTHON" -c "import twine" >/dev/null 2>&1; then
    echo "[ERROR] twine is not installed. Run: $PYTHON -m pip install twine" >&2
    exit 1
fi

if ! ls dist/* >/dev/null 2>&1; then
    echo "[ERROR] no distributions in ./dist - run ./build_pypi.sh first" >&2
    exit 1
fi

echo "[*] Checking distributions..."
"$PYTHON" -m twine check dist/*

echo "[*] Uploading to ${REPO_ARGS[*]:-pypi.org} ..."
"$PYTHON" -m twine upload ${REPO_ARGS[@]+"${REPO_ARGS[@]}"} dist/*

VERSION=$("$PYTHON" -c "import aic; print(aic.__version__)")
echo "[*] Done. aic-agent $VERSION"
echo "    https://pypi.org/project/aic-agent/$VERSION/"
