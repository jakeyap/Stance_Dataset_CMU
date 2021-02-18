#!/usr/bin/env bash
#@author: jakeyap on 20210218 8pm

for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
    echo Running batch $i
    python process_event_universe.py --segment=$i | tee process_event_universe_$i.log
done