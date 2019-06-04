/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   python_scorer.hpp
 * @brief  Oracle node scorer.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "python_scorer.hpp"

#include <iostream>
#include <utility>

#include "scip/struct_scip.h"

/**************************************************************************/
/* PythonScorer class */
/**************************************************************************/

namespace imilp {

/** Initialization */
void PythonScorer::Init() {
  feat_->ResetCache();
}

/** Deconstructor. */
void PythonScorer::DeInit() {}

/** Score a node. */
int PythonScorer::Compare(SCIP_NODE *node1, SCIP_NODE *node2) {
  // std::cout << scip_->tree->leaves << "\n";

  if (SCIPisInfinity(scip_, SCIPnodeGetLowerbound(node1))) {
    return +1;
  } else if (SCIPisInfinity(scip_, SCIPnodeGetLowerbound(node2))) {
    return -1;
  }

  std::vector<double> x1 = feat_->GetCachedFeatures(node1);
  std::vector<double> x2 = feat_->GetCachedFeatures(node2);
  // std::cout << x1.size() << " " << x2.size() << "\n";
  assert(x1.size() == feat_->GetNumFeatures());
  assert(x2.size() == feat_->GetNumFeatures());

  int result = model_->Predict(x1, x2);

  /* First node X1 was better than the second X2. */
  if (result == 1) {
    return -1;
  } else if (result == 0) {
    /* First node was not better */
    return +1;
  }

  /* Should only happen on error. */
  return 0;
}

}  // namespace imilp