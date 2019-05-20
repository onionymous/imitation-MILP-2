/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   feat.hpp
 * @brief  Node features to be passed to the ranking model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef FEAT_HPP
#define FEAT_HPP

#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>

#include "oracle.hpp"
#include "data_collector_base.hpp"
#include "scip/scip.h"
#include "scip/struct_sol.h"

namespace imilp {

/** Retrospective oracle data collector. */
class Feat {
 public:
  Feat() {}
  ~Feat() {};

  /** Get headers. */
  std::string GetHeaders();

  /** Get number of features. */
  int GetNumFeatures();

  /** Reset cache. */
  void ResetCache() { cache_.clear(); }

  /** Get from cache. */
  std::vector<double> GetCachedFeatures(SCIP_NODE *node);

  /** Compute features for a node */
  std::vector<double> ComputeFeatures(SCIP *scip, SCIP_NODE *node);

 private:
  /** Enum to index into feature vectors. */
  enum Feature {
    /** Depth of this node in the branch and bound tree */
    FEAT_DEPTH,

    /** Max depth of the branch and bound tree at which to gather data */
    FEAT_MAXDEPTH,

    /** Objective value of LP relaxation at root */
    FEAT_ROOTLPOBJ,

    /** Sum of coefficients of the objective */
    FEAT_SUMOBJCOEFF,

    /** Number of constraints of the problem */
    FEAT_NCONSTRS,

    /** nodeLowerBound / rootLowerBound */
    FEAT_NODELOWERBOUND,

    /** globalLowerBound / rootLowerBound */
    FEAT_GLOBALLOWERBOUND,

    /** global upper bound, normalized by lower bound on root */
    FEAT_GLOBALUPPERBOUND,

    /** 1 if normalized global upper bound is infinite; 0 otherwise */
    FEAT_GLOBALUPPERBOUNDINF,

    /** (nodeLower - globalLower) / (globalUpper - globalLower) */
    FEAT_RELATIVEBOUND,

    /** (globalUpperBound - globalLowerBound) / globalLowerBound */
    FEAT_GAP,

    /** 1 if gap b/w upper and lower bound is infinite; 0 otherwise */
    FEAT_GAPINF,

    /** Indicator variable for if node is child */
    FEAT_TYPE_CHILD,

    /** Indicator variable for if node is child */
    FEAT_TYPE_SIBLING,

    /** Indicator variable for if node is leaf */
    FEAT_TYPE_LEAF,

    /** nodeEstimate / rootLowerBound */
    FEAT_ESTIMATE,

    /** (nodeEstimate - globalLower) / (globalUpper - globalLower) */
    FEAT_RELATIVEESTIMATE,

    /** number of successive times a child was chosen as next node */
    FEAT_PLUNGEDEPTH,

    /** depth / maxdepth * 10 (maxdepth is constant) */
    FEAT_RELATIVEDEPTH,

    /** number of solutions found so far */
    FEAT_NSOLUTION,

    /** (bound of branch var) - (current value of var in solution) */
    FEAT_BRANCHVAR_BOUNDLPDIFF,

    /** (value of var at root) - (current value of var in solution) */
    FEAT_BRANCHVAR_ROOTLPDIFF,

    /** Pseudocost of branching variable */
    FEAT_BRANCHVAR_PSEUDOCOST,

    /** 1 if we prefer branching up; 0 otherwise */
    FEAT_BRANCHVAR_PRIO_UP,

    /** 1 if we prefer branching down; 0 otherwise */
    FEAT_BRANCHVAR_PRIO_DOWN,

    /** Avg number of inferences found after branching on this var */
    FEAT_BRANCHVAR_INF,

    /** Number of default features */
    N_FEATURES,
  };  /* enum Features */

  /** Map to translate features to strings. */
  static std::unordered_map<Feature, const char *> feature_names_;

  /** Cache. */
  std::unordered_map<long, std::vector<double>> cache_;

};

}  // namespace imilp

#endif