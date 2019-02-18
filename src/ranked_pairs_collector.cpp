/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ranked_pairs_collector.cpp
 * @brief  Data collector class that collects data for training a ranking model
           with retrospective knowledge of node optimality, and writes the
           collected data to a .csv file.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "ranked_pairs_collector.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>

// #include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include "scip/set.h"
#include "scip/struct_scip.h"
#include "scip/tree.h"
#include "scip/var.h"
#include "scip/stat.h"

/**************************************************************************/
/* RankedPairsCollector class */
/**************************************************************************/

namespace imilp {

/** Constructor. */
void RankedPairsCollector::Init() {
  /* Reset features. */
  feat_->ResetCache();

  namespace fs = boost::filesystem;

  /* If train file exists, open in append mode. */
  if (fs::is_regular_file(fs::path(train_filename_))) {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Appending train data to existing file: " << train_filename_
              << "\n";
    train_file_.open(train_filename_, std::ofstream::app);

    /* Otherwise, create a new file and write headers. */
  } else {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Creating new file for train data and writing headers: "
              << train_filename_ << "\n";
    train_file_.open(train_filename_, std::ofstream::out);
    // WriteHeaders();
    train_file_ << feat_->GetHeaders() << "\n";
  }

  /* If valid file exists, open in append mode. */
  if (fs::is_regular_file(fs::path(valid_filename_))) {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Appending validation data to existing file: "
              << valid_filename_ << "\n";
    valid_file_.open(valid_filename_, std::ofstream::app);

    /* Otherwise, create a new file and write headers. */
  } else {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Creating new file for validation data and writing headers: "
              << valid_filename_ << "\n";
    valid_file_.open(valid_filename_, std::ofstream::out);
    // WriteHeaders();
    valid_file_ << feat_->GetHeaders() << "\n";
  }

  /* Failed to load solutions. TODO */
  if (!oracle_->LoadSolutions()) {
    assert(0 && "[FATAL]: RankedPairsCollector: Oracle failed to initialize.");
  }
}

/** Deconstructor. */
void RankedPairsCollector::DeInit() {
  /* Close the output file. */
  train_file_.close();
  valid_file_.close();

  /* Free all the solutions. */
  oracle_->FreeSolutions();
}

/** Process the current SCIP state. */
void RankedPairsCollector::Process() {
  std::vector<std::pair<std::vector<SCIP_Real>, int>> opt_nodes;
  std::vector<std::vector<SCIP_Real>> non_opt_nodes;

  /* depth=1 => weight = 5; depth=0.6*maxdepth => weight = 1
     Use the deepest node in this training example to weight. */
  double weight = std::numeric_limits<double>::infinity();
  int max_depth = SCIPgetNBinVars(scip_) + SCIPgetNIntVars(scip_);

  /* Loop through all nodes in the priority queue. */
  for (int v = 2; v >= 0; --v) {
    SCIP_NODE* node;
    SCIP_NODE** open_nodes;
    int num_open_nodes = -1;

    SCIP_RETCODE retcode;

    switch (v) {
      case 2:
        retcode = SCIPgetChildren(scip_, &open_nodes, &num_open_nodes);
        assert(retcode == SCIP_OKAY);
        break;
      case 1:
        retcode = SCIPgetSiblings(scip_, &open_nodes, &num_open_nodes);
        assert(retcode == SCIP_OKAY);
        break;
      case 0:
        retcode = SCIPgetLeaves(scip_, &open_nodes, &num_open_nodes);
        assert(retcode == SCIP_OKAY);
        break;
      default:
        assert(0);
        break;
    }

    assert(num_open_nodes >= 0);

    /* Loop through each open node and compute features. */
    for (int n = num_open_nodes - 1; n >= 0 && !SCIPisStopped(scip_); --n) {
      node = open_nodes[n];

      if (!SCIPisInfinity(scip_, SCIPnodeGetLowerbound(node))) {
        /* Store nodes in the correct vector, depending on whether it is optimal
           with respect to the current solutions set or not. */
        int opt = oracle_->GetOptimality(node);

        if (opt > -1) {
          opt_nodes.push_back({feat_->ComputeFeatures(scip_, node), opt});
        } else {
          non_opt_nodes.push_back(feat_->ComputeFeatures(scip_, node));
        }
      }

      /* Update the weight for this training example. */
      weight = std::min(weight, 5 * exp(-(SCIPnodeGetDepth(node) - 1) /
                                        (0.6 * max_depth) * 1.61));
    }
  }

  // std::cout << opt_nodes.size() << " " << non_opt_nodes.size() << "\n";

  /* Generate a random number to determine whether this data point should go
     to the train or the validation set. */
  std::random_device rd; 
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);
  double p = dist(gen);

  /* Write out the data in pairs. */
  for (auto &opt_node : opt_nodes) {
    for (auto& non_opt_node : non_opt_nodes) {
      /* Write weight values */
      if (p < 0.2) {
        valid_file_ << weight << ",";
      } else {
        train_file_ << weight << ",";
      }

      /* Write X1 values */
      for (auto& feat : opt_node.first) {
        if (p < 0.2) {
          valid_file_ << feat << ",";
        } else {
          train_file_ << feat << ",";
        }
      }

      /* Write X2 values */
      for (auto& feat : non_opt_node) {
        if (p < 0.2) {
          valid_file_ << feat << ",";
        } else {
          train_file_ << feat << ",";
        }
      }

      /* Write training target (y) */
      if (p < 0.2) {
        valid_file_ << opt_node.second << "\n";
      } else {
        train_file_ << opt_node.second << "\n";
      }
    }
  }

  // feat_->ResetCache();
}

}  // namespace imilp