#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess


def parse_logs(log_file, mode):
    files = []
    stats = []

    mode == ""
    if mode == "nodes":
        match = "nodes  "
    elif mode == "time":
        match = "Total Time "
    elif mode == "primal":
        match = "Primal integral:"

    with open(log_file) as f:
        for line in f:
            l = line.strip()
            if "Problem name" in l and l.endswith(".lp"):
                files.append(l.split("/")[-1])
            elif match in l:
                stats.append(l.split(":")[-1].strip())

    for f, c in zip(files, stats):
        print("{},{}".format(f, c))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:")
        print("\t{} [log file] [nodes|time|primal]".format(sys.argv[0]))
    else:
        parse_logs(sys.argv[1], sys.argv[2].lower())
