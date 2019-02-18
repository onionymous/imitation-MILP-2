/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   oracle_scorer.hpp
 * @brief  Oracle node scorer.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "oracle_scorer.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// #include <boost/log/trivial.hpp>
#include "scip/struct_scip.h"

/**************************************************************************/
/* RankedPairsCollector class */
/**************************************************************************/

namespace imilp {

/** Constructor. */
void OracleScorer::Init() {
  /* Failed to load solutions. TODO */
  if (!oracle_->LoadSolutions()) {
    assert(0 && "[FATAL]: OracleScorer: Oracle failed to initialize.");
  }
}

/** Deconstructor. */
void OracleScorer::DeInit() {
  /* Free all the solutions. */
  oracle_->FreeSolutions();
}

/** Score a node. */
int OracleScorer::Compare(SCIP_NODE *node1, SCIP_NODE *node2) {
  int opt1 = oracle_->GetOptimality(node1);
  int opt2 = oracle_->GetOptimality(node2);

  if (opt1 > opt2) {
    return -1;
  } else if (opt2 > opt1) {
    return +1;
  }

  return 0;
}

}  // namespace imilp