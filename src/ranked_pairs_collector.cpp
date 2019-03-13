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

#define MAX_DIST_FROM_OPT 5

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
  if (fs::is_regular_file(fs::path(output_filename_))) {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Appending data to existing file: " << output_filename_
              << "\n";
    output_file_.open(output_filename_, std::ofstream::app);

    /* Otherwise, create a new file and write headers. */
  } else {
    std::cerr << "[INFO]: "
              << "RankedPairsCollector: "
              << "Creating new file for data and writing headers: "
              << output_filename_ << "\n";
    output_file_.open(output_filename_, std::ofstream::out);
    // WriteHeaders();
    output_file_ << feat_->GetHeaders() << "\n";
  }

  /* Failed to load solutions. TODO */
  if (!oracle_->LoadSolutions()) {
    assert(0 && "[FATAL]: RankedPairsCollector: Oracle failed to initialize.");
  }
}

/** Deconstructor. */
void RankedPairsCollector::DeInit() {
  /* Close the output file. */
  output_file_.close();

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
          /* Node is optimal. */
          opt_nodes.push_back({feat_->ComputeFeatures(scip_, node), opt});

        } else {
          /* Node is non-optimal, append it if it is within cutoff from last
             optimal node. */
          int dist_from_opt = oracle_->GetDistanceFromOpt(node);
          if (dist_from_opt <= MAX_DIST_FROM_OPT) {
            /* compute and append */
            non_opt_nodes.push_back(feat_->ComputeFeatures(scip_, node));
          } else {
            /* compute only. */
            feat_->ComputeFeatures(scip_, node);
          }
        }
      }

      /* Update the weight for this training example. */
      weight = std::min(weight, 5 * exp(-(SCIPnodeGetDepth(node) - 1) /
                                        (0.6 * max_depth) * 1.61));
    }
  }

  // std::cout << opt_nodes.size() << " " << non_opt_nodes.size() << "\n";

  /* Generate a random number to determine whether this data point should be 
     flipped. */
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(0.0, 1.0);

  /* Probability to write this data point. */
  double prob_write = dist(gen);

  /* Probability to flip this output. */
  double prob_flip_pair = dist(gen);

  /* Write out the data in pairs. */
  for (auto& opt_node : opt_nodes) {
    for (auto& non_opt_node : non_opt_nodes) {
      if (prob_write <= sample_rate_) {
        /* Write weight values */
        output_file_ << weight << ",";

        if (prob_flip_pair <= 0.5) {
          /* Write X1 values */
          for (auto& feat : opt_node.first) {
            output_file_ << feat << ",";
          }

          /* Write X2 values */
          for (auto& feat : non_opt_node) {
            output_file_ << feat << ",";
          }

          /* Write training target (y), that X1 better than X2 */
          output_file_ << 1 << "\n";

        } else {
          /* Write X2 values */
          for (auto& feat : non_opt_node) {
            output_file_ << feat << ",";
          }

          /* Write X1 values */
          for (auto& feat : opt_node.first) {
            output_file_ << feat << ",";
          }

          /* Write training target (y), that X1 not better than X2 */
          output_file_ << 0 << "\n";
        }
      }
    }
  }

  // feat_->ResetCache();
}

}  // namespace imilp