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
                       bool is_append, Oracle *oracle, Feat *feat,
                       double sample_rate)
      : DataCollectorBase(scip),
        output_filename_(output_filename),
        is_append_(is_append),
        oracle_(oracle),
        feat_(feat),
        sample_rate_(sample_rate) {}
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

  /** Append to existing file? */
  bool is_append_;

  /** Retrospective oracle. */
  Oracle *oracle_;

  /** Features. */
  Feat *feat_;

  /** Sampling rate for data collection output to files. (0.0 to write nothing,
      1.0 to write everything). */
  double sample_rate_;
};

}  // namespace imilp

#endif