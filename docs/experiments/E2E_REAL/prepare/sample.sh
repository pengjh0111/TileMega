#!/bin/bash
PID=$1; N=$2; SLEEP=$3; LOG=$4
for i in $(seq 1 $N); do
  ps -p $PID >/dev/null 2>&1 || { echo "gone"; break; }
  imports=$(grep -c 'aten.embedding.default' $LOG)
  bt=$(gdb -p $PID -batch -ex "bt 40" 2>/dev/null | grep -E '^#')
  deep=$(echo "$bt" | grep -oE '(tilemega::[A-Za-z0-9_:]+|isl_[a-z_0-9]+)' | head -1)
  mid=$(echo "$bt" | grep -oE 'tilemega::(frontend|analysis|solver|dialect|codegen)::[A-Za-z0-9_:]+' | head -3 | tr '\n' '|')
  echo -e "$(date +%s)\t$imports\t$deep\t$mid"
  sleep $SLEEP
done
