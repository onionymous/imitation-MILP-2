/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   feat_computer_collector.cpp
 * @brief  Data collector class that collects data for use with ranked pairs
           model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "feat_computer_collector.hpp"

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
/* FeatComputerCollector class */
/**************************************************************************/

namespace imilp {

/** Constructor. */
void FeatComputerCollector::Init() {
  /* Reset features. */
  feat_->ResetCache();
}

/** Deconstructor. */
void FeatComputerCollector::DeInit() {}

/** Process the current SCIP state. */
void FeatComputerCollector::Process() {
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
        /* compute only. */
        feat_->ComputeFeatures(scip_, node);
      }
    }
  }
}

}  // namespace imilp