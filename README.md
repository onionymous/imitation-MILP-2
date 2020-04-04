# Imitation MILP

This is a repository that demonstrates how to use DAgger to accelerate mixed-integer linear programming. It is based on the popular free and open-source MILP solver, [SCIP](https://scip.zib.de/). The ranking model used is RankNet.

## Installation and compilation

### Prerequisites

* SCIP
* Gurobi (or other LP solver)
* Python 3 with numpy, scikit-learn, Pytorch
* g++ 
* libboost
* libtorch

This repository requires an installation of [SCIP](https://scip.zib.de/). We use [Gurobi](https://www.gurobi.com/) as the LP solver, although other LP solvers can be used.

Ensure that the environment variable `SCIPDIR` is set to your SCIP installation, since it is referenced in the Makefile.

### Compilation

```
git clone https://github.com/onionymous/imitation-MILP-2
cd imitation-MILP-2
make LPS=grb ZIMPL=false
```

The main executable should be at `bin/imilp` after compilation.

## Running

The problems used for training, validation and testing should be in `.lp` format. Solutions for the problem are in a `solutions` subdirectory of the folder containing the problems. Each problem has a folder with the same name as the problem (without the `.lp` suffix), with the solutions for that problem in the folder. (e.g. if the `test/` folder is a directory containing a problem named `00000.lp`, `test/solutions/00000/00000.sol` is the solution to that problem.)

Please see the problems in the `data/` folder (e.g. `data/hybrid_bids/bids_500/test`) for an example of the data format and folder structure. 

### Getting oracle trajectories

The initial model is trained on the oracle trajectories. To collect this data, run with `oracle` as the mode (`-M`) flag:
```
bin/imilp -M oracle -p [problems folder] -o [folder to save trajectories data]
```

It is recommended to save the output of the run as a training log, e.g.
```
bin/imilp -M oracle -p data/hybrid_bids/bids_500/valid/ -o data/hybrid_bids/bids_500/valid/oracle 2>&1 | tee bids_500_valid_oracle.log
```

Since the oracle trajectories require an existing solution in order to get the retrospective best node selections, if a valid solution for a problem is not detected (in the `solutions/[problem_name]` folder), default SCIP will be invoked to obtain an initial solution.


### Training the model
`utils/train_model.py` demonstrates how to train the RankNet model, using the data collected during the solving process. Modify the driver code for your training data, e.g.:

```
train_dirs = ["data/hybrid_bids/bids_500/train/oracle",
              "data/hybrid_bids/bids_500/train/iter1"]
valid_dirs = ["data/hybrid_bids/bids_500/valid/oracle",
              "data/hybrid_bids/bids_500/valid/iter1"]

m = RankNet("models/bids_500-1.pt", 26, "")
m.train(train_dirs, valid_dirs, 100, 32)
```

* `RankNet(model_save_path, n_features, previous_model)`
* `RankNet.train(train_dirs, valid_dirs, n_iters, batch_size)`


### Running SCIP with the model
Run with `model` as the mode flag:
```
bin/imilp -M model -p [problems folder] -o [folder to save trajectories data] -m [model path]
```

The specified model should be a valid [TorchScript](https://pytorch.org/docs/stable/jit.html) model.

Again, it is recommended to save the solving logs to extract statistics later on:
```
bin/imilp -M model -p data/hybrid_bids/bids_500/test -o data/hybrid_bids/bids_500/test/iter1 -m models/bids_500-0.pt 2>&1 | tee bids_500_test_iter1.log
```

### DAgger loop
This process can be repeated in a loop; i.e. training a model with the initial oracle trajectories, then collecting data during the solving process, using the oracle model, then training another model with the new data collected, until the desired number of iterations of DAgger is completed.

## Utilities

The `utils/` folder also contains some other scripts that may be useful.

* `grb_to_scip.py` converts files in Gurobi's LP format to a format accepted by SCIP.
* `parse_logs.py` parses important stats of problems from the SCIP solving logs. The usage is `python3 utils/parse_logs.py [log file] [mode=nodes|primal|time]`. This script takes in a solving log and outputs each problem name and the associated statistic, in comma-separated format. For example, to get the number of nodes examined in a solving run:
```
490000.lp,16 (8 internal, 8 leaves)
00000.lp,5402 (2701 internal, 2701 leaves)
10000.lp,11 (5 internal, 6 leaves)
20000.lp,29 (14 internal, 15 leaves)
30000.lp,24198 (12100 internal, 12098 leaves)
...
```