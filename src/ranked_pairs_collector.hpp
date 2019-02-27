/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ranked_pairs_collector.hpp
 * @brief  Data collector class that collects data for training a ranking model
           with retrospective knowledge of node optimality, and writes the
           collected data to a .csv file.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef RANKED_PAIRS_COLLECTOR_HPP
#define RANKED_PAIRS_COLLECTOR_HPP

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

/** Retrospective oracle data collector. */
class RankedPairsCollector : public DataCollectorBase {
 public:
  RankedPairsCollector(SCIP *scip, const std::string &output_filename,
                       Oracle *oracle, 
                       Feat *feat)
      : DataCollectorBase(scip),
        output_filename_(output_filename),
        oracle_(oracle),
        feat_(feat) {}
  ~RankedPairsCollector() {};

  /** Process the current SCIP instance. */
  void Init() override;
  void DeInit() override;
  void Process() override;

 private:
  /** Output file name. */
  std::string output_filename_;

  /** File stream to save collected data to. */
  std::ofstream output_file_;

  /** Retrospective oracle. */
  Oracle *oracle_;

  /** Features. */
  Feat *feat_;
};

}  // namespace imilp

#endif