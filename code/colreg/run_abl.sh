#!/bin/bash
cd ~/stripe_seg/colreg
PY=~/miniforge3/envs/stripe/bin/python
run() { n=$1; shift; $PY train3.py --name $n --iters 8000 --workers 6 "$@" > train_$n.log 2>&1; }
export -f run; export PY
printf '%s\n' \
 "abl_base_s0 --seed 0" \
 "abl_pre --seed 0 --weights imagenet" \
 "abl_seg --seed 0 --head seg" \
 "abl_fullres --seed 0 --arch fullres" \
 "abl_base_s1 --seed 1" \
 "abl_synth --seed 0 --p_synth 1.0" \
 "abl_real --seed 0 --p_synth 0.0" | xargs -P 3 -I{} bash -c 'run {}'
echo done > abl_done.txt
