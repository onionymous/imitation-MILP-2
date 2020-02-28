#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess


def parse_logs(log_file):
    files = []
    counts = []

    with open(log_file) as f:
        for line in f:
            l = line.strip()
            if "Problem name" in l and l.endswith(".lp"):
                files.append(l.split("/")[-1])
            elif "nodes  " in l:
                counts.append(l.split(":")[-1].strip())

    for f, c in zip(files, counts):
        print("{},{}".format(f, c))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage:")
        print("\t{} [log file]".format(sys.argv[0]))
    else:
        parse_logs(sys.argv[1])
