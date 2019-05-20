/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   retrospective_oracle.hpp
 * @brief  Data collector class that collects data for training a ranking model
           with retrospective knowledge of node optimality, and writes the
           collected data to a .csv file.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef ORACLE_HPP
#define ORACLE_HPP

#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>

#include "scip/scip.h"
#include "scip/struct_sol.h"

namespace imilp {

/** Type definition to distinguish node IDs from regular long integers. */
typedef SCIP_Longint NodeId;

/** Retrospective oracle data collector. */
class Oracle {
 public:
  Oracle(SCIP *scip, const std::string &solutions_dir)
      : solutions_dir_(solutions_dir), scip_(scip) {}
  ~Oracle();

  /** Read all solutions in the given directory. */
  bool LoadSolutions();

  /** Free all solutions. */
  void FreeSolutions();

  /** Check optimality of a node with respect to the current list of known
      solutions. */
  int GetOptimality(SCIP_NODE *node);

  /** Get distance of a node from the last optimal solution. */
  int GetDistanceFromOpt(SCIP_NODE *node);

  /** Get optimal objective value. */
  double GetOptObjectiveValue();

 private:
  /** Path to soliutins directory. */
  std::string solutions_dir_;

  /** SCIP object. */
  SCIP *scip_;

  /** Current list of solutions. */
  std::vector<SCIP_SOL*> solutions_;

  /** Cache for optimality. */
  std::unordered_map<NodeId, int> node_opt_cache_;

  /** Cache for distance from last optimal node. */
  std::unordered_map<NodeId, int> opt_dist_cache_; 

};

}  // namespace imilp

#endif