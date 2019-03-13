/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   feat_computer_collector.hpp
 * @brief  Data collector class that collects data for use with ranked pairs
           model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef FEAT_COMPUTER_COLLECTOR_HPP
#define FEAT_COMPUTER_COLLECTOR_HPP

#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>

#include "oracle.hpp"
#include "feat.hpp"
#include "data_collector_base.hpp"
#include "scip/scip.h"
#include "scip/struct_sol.h"

namespace imilp {

/** Feature computer data collector. */
class FeatComputerCollector : public DataCollectorBase {
 public:
  FeatComputerCollector(SCIP *scip, Feat *feat)
      : DataCollectorBase(scip),
        feat_(feat) {}
  ~FeatComputerCollector() {};

  /** Process the current SCIP instance. */
  void Init() override;
  void DeInit() override;
  void Process() override;

 private:
  /** Features. */
  Feat *feat_;
};

}  // namespace imilp

#endif