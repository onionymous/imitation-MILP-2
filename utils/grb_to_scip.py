#!/usr/bin/env python3

import sys
import os
from pathlib import Path


def convert_file(old_file_path, new_file_path):
    with open(old_file_path, "r") as old_file, open(new_file_path, "w") as new_file:
        for line in old_file:
            if line.startswith("#"):
                if line.startswith("# Objective value = "):
                    obj_val = line.strip().split(" = ")[1]
                    new_file.write("objective value: {}\n".format(obj_val))
            else:
                new_file.write(line)
    print("Converted file: {}".format(new_file_path))


def process_dir(old_path, new_path):
    if not os.path.exists(new_path):
        print("Creating folder: {}".format(new_path))
        os.mkdir(new_path)

    old_path = Path(old_path)
    new_path = Path(new_path)

    for entry in old_path.iterdir():
        if entry.is_file():
            old_file = entry
            sol_path = new_path / entry.stem
            if not os.path.exists(sol_path):
                os.mkdir(sol_path)
            new_file = sol_path / entry.name
            convert_file(old_file, new_file)
            # print(entry)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:")
        print("\t{} [folder of files to convert] [folder to save converted files]".format(
            sys.argv[0]))
    else:
        process_dir(sys.argv[1], sys.argv[2])
