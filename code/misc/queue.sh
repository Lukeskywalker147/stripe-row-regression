#!/bin/bash
# usage: queue.sh "arch:encoder:seed" ...
cd ~/stripe_seg/stripe_unet
PY=~/miniforge3/envs/stripe/bin/python
for job in "$@"; do
  IFS=: read arch enc seed <<< "$job"
  name=scan3_${arch}_${enc}_s${seed}
  $PY train.py --config config.yaml --set arch=$arch encoder=$enc seed=$seed in_channels=3 data_root=~/stripe_seg/data_scan_v3 out_dir=~/stripe_seg/runs/$name > ../runs/$name.log 2>&1
done
