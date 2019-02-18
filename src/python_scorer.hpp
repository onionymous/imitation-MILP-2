/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   python_scorer.hpp
 * @brief  Python node scorer.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef PYTHON_SCORER_HPP
#define PYTHON_SCORER_HPP

#include "scorer_base.hpp"
#include "scip/scip.h"
#include "scip/struct_sol.h"

#include "ranknet_model.hpp"
#include "feat.hpp"

namespace imilp {

/** Python scorer. */
class PythonScorer : public ScorerBase {
 public:
  PythonScorer(SCIP *scip, RankNetModel *model, Feat *feat)
      : ScorerBase(scip), model_(model), feat_(feat) {}
  ~PythonScorer() {};

  /** Process the current SCIP instance. */
  void Init() override;
  void DeInit() override;
  int Compare(SCIP_NODE *node1, SCIP_NODE *node2) override;

 private:
  RankNetModel *model_;
  Feat *feat_;
};

}  // namespace imilp

#endif