#!/bin/bash
# usage: ./mon.sh [run_name ...]   default: scan2_r18 scan2_mbv3   (refresh every 5 s, Ctrl-C to quit)
RUNS=${@:-scan2_r18 scan2_mbv3}
CMD=""
for r in $RUNS; do
  CMD="$CMD echo \"== $r\"; tail -n 3 $HOME/stripe_seg/runs/$r.log | cut -c1-210; echo \"best: \$(tr -d \"\\n \" < $HOME/stripe_seg/runs/$r/best.json 2>/dev/null)\"; echo;"
done
watch -n 5 "$CMD nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader"
