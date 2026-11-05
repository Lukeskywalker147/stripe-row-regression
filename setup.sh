#!/bin/bash
# Link data next to the code, as the scripts expect (colreg/bg, colreg/labels), and expose data_scan_v3 at ~/stripe_seg.
set -e
cd "$(dirname "$0")"
ln -sfn ../../data/real_train/bg code/colreg/bg
ln -sfn ../../data/real_train/labels code/colreg/labels
mkdir -p ~/stripe_seg && ln -sfn "$PWD/data/real_train/data_scan_v3" ~/stripe_seg/data_scan_v3
mkdir -p code/colreg/runs
for f in release_assets/*.pt; do n=$(basename "$f" .pt); mkdir -p code/colreg/runs/$n; cp -n "$f" code/colreg/runs/$n/last.pt; [ -f logs/$n/meta.json ] && cp -n logs/$n/meta.json code/colreg/runs/$n/; done
echo "done"
