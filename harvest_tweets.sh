#!/usr/bin/env bash
#@author: jakeyap on 20210218 8pm

# for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# for i in 2 8 11 12 13 15 18
# for i in 2 11 12 13 15 18
for i in 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37
do
    echo Running batch $i
    python process_event_universe.py --segment=$i | tee process_event_universe_$i.log
done
