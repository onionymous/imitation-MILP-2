/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   feat.cpp
 * @brief  Node features to be passed to the ranking model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "feat.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// #include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include "scip/set.h"
#include "scip/stat.h"
#include "scip/struct_scip.h"
#include "scip/tree.h"
#include "scip/var.h"

namespace imilp {

/** Features enum to string map. */
std::unordered_map<Feat::Feature, const char*> Feat::feature_names_ = {
    {FEAT_DEPTH, "FEAT_DEPTH"},
    {FEAT_MAXDEPTH, "FEAT_MAXDEPTH"},
    {FEAT_ROOTLPOBJ, "FEAT_ROOTLPOBJ"},
    {FEAT_SUMOBJCOEFF, "FEAT_SUMOBJCOEFF"},
    {FEAT_NCONSTRS, "FEAT_NCONSTRS"},
    {FEAT_NODELOWERBOUND, "FEAT_NODELOWERBOUND"},
    {FEAT_GLOBALLOWERBOUND, "FEAT_GLOBALLOWERBOUND"},
    {FEAT_GLOBALUPPERBOUND, "FEAT_GLOBALUPPERBOUND"},
    {FEAT_GLOBALUPPERBOUNDINF, "FEAT_GLOBALUPPERBOUNDINF"},
    {FEAT_RELATIVEBOUND, "FEAT_RELATIVEBOUND"},
    {FEAT_GAP, "FEAT_GAP"},
    {FEAT_GAPINF, "FEAT_GAPINF"},
    {FEAT_TYPE_CHILD, "FEAT_TYPE_CHILD"},
    {FEAT_TYPE_SIBLING, "FEAT_TYPE_SIBLING"},
    {FEAT_TYPE_LEAF, "FEAT_TYPE_LEAF"},
    {FEAT_ESTIMATE, "FEAT_ESTIMATE"},
    {FEAT_RELATIVEESTIMATE, "FEAT_RELATIVEESTIMATE"},
    {FEAT_PLUNGEDEPTH, "FEAT_PLUNGEDEPTH"},
    {FEAT_RELATIVEDEPTH, "FEAT_RELATIVEDEPTH"},
    {FEAT_NSOLUTION, "FEAT_NSOLUTION"},
    {FEAT_BRANCHVAR_BOUNDLPDIFF, "FEAT_BRANCHVAR_BOUNDLPDIFF"},
    {FEAT_BRANCHVAR_ROOTLPDIFF, "FEAT_BRANCHVAR_ROOTLPDIFF"},
    {FEAT_BRANCHVAR_PSEUDOCOST, "FEAT_BRANCHVAR_PSEUDOCOST"},
    {FEAT_BRANCHVAR_PRIO_UP, "FEAT_BRANCHVAR_PRIO_UP"},
    {FEAT_BRANCHVAR_PRIO_DOWN, "FEAT_BRANCHVAR_PRIO_DOWN"},
    {FEAT_BRANCHVAR_INF, "FEAT_BRANCHVAR_INF"},
    {N_FEATURES, "N_FEATURES"},
};

/** Get number of features. */
int Feat::GetNumFeatures() {
  return N_FEATURES;
}

/** Write headers to file. */
std::string Feat::GetHeaders() {
  std::string headers = "weight,";

  /* Columns for X1 features. */
  for (int i = 0; i < N_FEATURES; ++i) {
    headers +=
        (std::string(feature_names_[static_cast<Feature>(i)]) + "1" + ",");
  }

  /* Columns for X2 features. */
  for (int i = 0; i < N_FEATURES; ++i) {
    headers +=
        (std::string(feature_names_[static_cast<Feature>(i)]) + "2" + ",");
  }

  /* y or training target. */
  headers += "target";

  return headers;
}

/** Get cached features for the specified node. */
std::vector<double> Feat::GetCachedFeatures(SCIP_NODE* node) {
  long node_id = SCIPnodeGetNumber(node);

  assert(cache_.find(node_id) != cache_.end());

  return cache_[node_id];
}

/** Compute features for the specified node. */
std::vector<double> Feat::ComputeFeatures(SCIP* scip, SCIP_NODE* node) {
  long node_id = SCIPnodeGetNumber(node);

  // if (cache_.find(node_id) != cache_.end()) {
  //   return cache_[node_id];
  // }

  std::vector<SCIP_Real> vals(N_FEATURES, 0.0);

  /* Make sure the node is non-null */
  assert(node != NULL);

  /* Make sure we the node is non-null */
  int depth = SCIPnodeGetDepth(node);
  int maxdepth = SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip);
  assert(depth != 0);
  assert(maxdepth != 0);

  /**************************************************************************/
  /*                        CALCULATING BOUNDS                              */
  /**************************************************************************/
  /* Info about lower bound globally, at current node, and at root. */
  SCIP_Real lowerbound = SCIPgetLowerbound(scip);
  SCIP_Real nodelowerbound = SCIPnodeGetLowerbound(node);
  SCIP_Real rootlowerbound = REALABS(SCIPgetLowerboundRoot(scip));
  assert(!SCIPisInfinity(scip, rootlowerbound));
  assert(!SCIPisInfinity(scip, lowerbound));

  if (SCIPisZero(scip, rootlowerbound)) rootlowerbound = 0.0001;
  vals[FEAT_NODELOWERBOUND] = nodelowerbound / rootlowerbound;
  vals[FEAT_GLOBALLOWERBOUND] = lowerbound / rootlowerbound;

  /* Global upper bound info. */
  SCIP_Bool upperboundinf;
  SCIP_Real upperbound = SCIPgetUpperbound(scip);
  if (SCIPisInfinity(scip, upperbound) || SCIPisInfinity(scip, -upperbound)) {
    upperboundinf = TRUE;
    vals[FEAT_GLOBALUPPERBOUNDINF] = 1;
    // if upper bound is infinite, use only 20% of gap as upper bound
    upperbound = lowerbound + 0.2 * (upperbound - lowerbound);
  } else {
    upperboundinf = FALSE;
    vals[FEAT_GLOBALUPPERBOUND] = upperbound / rootlowerbound;
  }

  /* Relative difference between node and global lower bound */
  if (!SCIPisEQ(scip, upperbound, lowerbound)) {
    vals[FEAT_RELATIVEBOUND] =
        (nodelowerbound - lowerbound) / (upperbound - lowerbound);
  }

  /* Calculate duality gap. */
  if (SCIPisEQ(scip, upperbound, lowerbound))
    vals[FEAT_GAP] = 0;
  else if (SCIPisZero(scip, lowerbound) || upperboundinf)
    vals[FEAT_GAPINF] = 1;
  else
    vals[FEAT_GAP] = (upperbound - lowerbound) / REALABS(lowerbound);

  /**************************************************************************/
  /*                       DETERMINING NODE TYPE                            */
  /**************************************************************************/
  SCIP_NODETYPE nodetype = SCIPnodeGetType(node);
  switch (nodetype) {
    case SCIP_NODETYPE_CHILD:
      vals[FEAT_TYPE_CHILD] = 1.0;
      break;
    case SCIP_NODETYPE_SIBLING:
      vals[FEAT_TYPE_SIBLING] = 1.0;
      break;
    case SCIP_NODETYPE_LEAF:
      vals[FEAT_TYPE_LEAF] = 1.0;
      break;
    default:
      break;
  }

  /**************************************************************************/
  /*                      MISCELLANEOUS FEATURES                            */
  /**************************************************************************/
  SCIP_Real est = SCIPnodeGetEstimate(node);
  vals[FEAT_ESTIMATE] = est / rootlowerbound;
  if (!SCIPisEQ(scip, upperbound, lowerbound)) {
    vals[FEAT_RELATIVEESTIMATE] =
        (est - lowerbound) / (upperbound - lowerbound);
  }

  vals[FEAT_DEPTH] = depth;
  vals[FEAT_MAXDEPTH] = maxdepth;
  vals[FEAT_PLUNGEDEPTH] = SCIPgetPlungeDepth(scip);
  vals[FEAT_RELATIVEDEPTH] =
      10.0 * static_cast<SCIP_Real>(depth) / static_cast<SCIP_Real>(maxdepth);

  vals[FEAT_NSOLUTION] = SCIPgetNSolsFound(scip);

  /**************************************************************************/
  /*                    BRANCHING VARIABLE FEATURES                         */
  /**************************************************************************/
  /* Get info about domain changes and bound changes */
  SCIP_DOMCHG* domchg = SCIPnodeGetDomchg(node);
  SCIP_BOUNDCHG* boundchgs = SCIPdomchgGetBoundchg(domchg, 0);
  assert(boundchgs != NULL);
  assert(SCIPboundchgGetBoundchgtype(boundchgs) == SCIP_BOUNDCHGTYPE_BRANCHING);

  /* Info about what variable we branched on. Only support branching on one
   * variable at a time right now. */
  SCIP_VAR* branchvar = SCIPboundchgGetVar(boundchgs);
  SCIP_Real varsol =
      SCIPvarGetSol(branchvar, SCIPtreeHasFocusNodeLP(scip->tree));
  SCIP_Real varrootsol = SCIPvarGetRootSol(branchvar);

  /* Info about bounds on the branching variable. */
  SCIP_Real branchbound = SCIPboundchgGetNewbound(boundchgs);
  SCIP_BOUNDTYPE boundtype = SCIPboundchgGetBoundtype(boundchgs);

  /* Features relating to bounds on branching variable */
  vals[FEAT_BRANCHVAR_BOUNDLPDIFF] = branchbound - varsol;
  vals[FEAT_BRANCHVAR_ROOTLPDIFF] = varrootsol - varsol;
  vals[FEAT_BRANCHVAR_PSEUDOCOST] =
      SCIPvarGetPseudocost(branchvar, scip->stat, branchbound - varsol);

  /* Preferred branching direction */
  SCIP_BRANCHDIR branchdirpreferred =
      static_cast<SCIP_BRANCHDIR>(SCIPvarGetBranchDirection(branchvar));
  if (branchdirpreferred == SCIP_BRANCHDIR_DOWNWARDS)
    vals[FEAT_BRANCHVAR_PRIO_DOWN] = 1;
  else if (branchdirpreferred == SCIP_BRANCHDIR_UPWARDS)
    vals[FEAT_BRANCHVAR_PRIO_UP] = 1;

  /* Average number of inferences */
  SCIP_Real avgInf = boundtype == SCIP_BOUNDTYPE_LOWER
                         ? SCIPvarGetAvgInferences(branchvar, scip->stat,
                                                   SCIP_BRANCHDIR_UPWARDS)
                         : SCIPvarGetAvgInferences(branchvar, scip->stat,
                                                   SCIP_BRANCHDIR_DOWNWARDS);
  vals[FEAT_BRANCHVAR_INF] = avgInf / static_cast<SCIP_Real>(maxdepth);

  /* Cache and return */
  cache_[node_id] = vals;
  return vals;
}

}  // namespace imilp