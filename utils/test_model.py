#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess


def test_problem(model_path, problem_path, save_path):
    f = open(save_path, "a+")

    print("Testing problem: {} (default SCIP)".format(problem_path))
    f.write("> Default SCIP:\n")
    f.flush()
    subprocess.call(["bin/imilp", "-p", problem_path], stdout=f)

    f.write("\n\n> With model:\n")
    f.flush()
    print("Testing problem: {} (model)".format(problem_path))
    subprocess.call(["bin/imilp", "-p", problem_path,
                     "-m", model_path], stdout=f)


def test_folder(model_path, problems_dir, save_dir):
    if not os.path.exists(model_path):
        print("Could not load model: {} (file does not exist)".format(model_path))
        sys.exit(1)

    if not os.path.exists(problems_dir):
        print("Problems directory does not exist: {}".format(problems_dir))
        sys.exit(1)

    if not os.path.exists(save_dir):
        print("Creating folder: {}".format(save_dir))
        os.mkdir(save_dir)

    problems_path = Path(problems_dir)
    save_path = Path(save_dir)

    print("Testing model: {} on folder: {}".format(model_path, problems_dir))
    for problem in problems_path.iterdir():
        if problem.is_file() and problem.suffix == '.lp':
            save_file = save_path / (problem.stem + '_test.txt')
            test_problem(model_path, problem.absolute().as_posix(),
                         save_file.absolute().as_posix())

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage:")
        print("\t{} [model_file] [folder of problems] [folder to save results]".format(
            sys.argv[0]))
    else:
        test_folder(sys.argv[1], sys.argv[2], sys.argv[3])
