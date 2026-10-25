#!/usr/bin/env bash
# Bump the pinned llama.cpp release: ./Scripts/update-llama.sh b11849
set -euo pipefail

tag="${1:?usage: update-llama.sh <release-tag, e.g. b11849>}"
root="$(cd "$(dirname "$0")/.." && pwd)"
url="https://github.com/ggml-org/llama.cpp/releases/download/${tag}/llama-${tag}-xcframework.zip"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "Downloading ${url}"
curl -fsSL -o "${work}/llama.zip" "$url"
sum="$(shasum -a 256 "${work}/llama.zip" | awk '{print $1}')"

python3 -I - "$root/Package.swift" "$tag" "$sum" <<'PY'
import re, sys
path, tag, sum_ = sys.argv[1:]
s = open(path).read()
s, n1 = re.subn(r'releases/download/b\d+/llama-b\d+-xcframework\.zip',
                f'releases/download/{tag}/llama-{tag}-xcframework.zip', s)
s, n2 = re.subn(r'checksum: "[0-9a-f]{64}"', f'checksum: "{sum_}"', s)
if n1 != 1 or n2 != 1:
    sys.exit("Package.swift binaryTarget not found in expected form")
open(path, "w").write(s)
PY

echo "Pinned ${tag} (sha256 ${sum})"
echo "Next: swift build && swift test, check the minos in 'vtool -show-build', update CHANGELOG.md, and run loci-cli eval."
