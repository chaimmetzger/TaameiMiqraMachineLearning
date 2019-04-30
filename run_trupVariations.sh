#!/bin/sh
for i in $(seq 254 255); #pico run_trupVariations.sh ./run_trupVariations.sh
do
  echo "obase=2;$i" | bc | awk '{printf "python trupsequencer14.py -partial -emet -%08d\n", $0}' | bash
done
