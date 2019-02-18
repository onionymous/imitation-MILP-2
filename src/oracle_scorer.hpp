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

#ifndef ORACLE_SCORER_HPP
#define ORACLE_SCORER_HPP

#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>

#include "oracle.hpp"
#include "scorer_base.hpp"
#include "scip/scip.h"
#include "scip/struct_sol.h"

namespace imilp {

/** Oracle scorer. */
class OracleScorer : public ScorerBase {
 public:
  OracleScorer(SCIP *scip, Oracle *oracle)
      : ScorerBase(scip),
        oracle_(oracle) {}
  ~OracleScorer() {};

  /** Process the current SCIP instance. */
  void Init() override;
  void DeInit() override;
  int Compare(SCIP_NODE *node1, SCIP_NODE *node2) override;

 private:
  /** Retrospective oracle. */
  Oracle *oracle_;
};

}  // namespace imilp

#endif