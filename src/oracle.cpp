/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   retrospective_oracle.cpp
 * @brief  Data collector class that collects data for training a ranking model
           with retrospective knowledge of node optimality, and writes the
           collected data to a .csv file.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "oracle.hpp"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

// #include <boost/log/trivial.hpp>
#include <boost/filesystem.hpp>
#include "scip/set.h"
#include "scip/struct_scip.h"
#include "scip/tree.h"
#include "scip/var.h"
#include "scip/stat.h"

/**************************************************************************/
/* Oracle class */
/**************************************************************************/

namespace imilp {

/** Destructor. */
Oracle::~Oracle() {
/* Free the solution objects. */
  FreeSolutions();
}

/** Destructor. */
void Oracle::FreeSolutions() {
/* Free the solution objects. */
  for (unsigned i = 0; i < solutions_.size(); ++i) {
    SCIPfreeSol(scip_, &(solutions_[i]));
  }
  solutions_.clear();
}

/** Load optimal solutions. Returns true on success, false on failure. */
bool Oracle::LoadSolutions() {
  bool success = true;

  /* If already loaded, just return. */
  if (!solutions_.empty()) {
    return success;
  }

  /* Loop through each file in the solutions directory and read a solution. */
  namespace fs = boost::filesystem;
  for (auto& sol_path : fs::directory_iterator(fs::path(solutions_dir_))) {
    SCIP_Bool partial, error;
    SCIP_SOL *sol;

    SCIPcreateSol(scip_, &sol, NULL);
    SCIP_RETCODE retcode = SCIPreadSolFile(
        scip_, sol_path.path().string().c_str(), sol, FALSE, &partial, &error);
    
    if (retcode != SCIP_OKAY) {
      std::cerr << "[ERROR]: " << "Oracle: "
          << "Could not create solution for file: " << sol_path.path().string();
      success = false;
      break;
    }

    std::cerr << "[INFO]: " << "Oracle: "
              << "Loaded solution file: " << sol_path.path().string() << "\n";
    solutions_.push_back(sol);
  }

  /* Sort solutions by objective value from least to greatest. */
  std::sort(solutions_.begin(), solutions_.end(),
            [](SCIP_SOL* sol1, SCIP_SOL* sol2) -> bool {
              return (sol1->obj < sol2->obj);
            });

  /* If maximizing, least to greatest objective = least to most optimal.
   * However, we want to sort from least to most optimal regardless of
   * whether we are minimizing or maximizing, so reverse if minimizing. */
  if (SCIPgetObjsense(scip_) == SCIP_OBJSENSE_MINIMIZE) {
    std::reverse(solutions_.begin(), solutions_.end());
  }

  /* If something failed, free all the solutions. */
  if (!success) {
    /* Free the solution objects. */
    for (unsigned i = 0; i < solutions_.size(); ++i) {
      SCIPfreeSol(scip_, &(solutions_[i]));
    }
    solutions_.clear();
  }

  return success;
}

/** Check optimality of a node with respect to the list of current known
    solutions. */
int Oracle::GetOptimality(SCIP_NODE* node) {
  NodeId node_id = SCIPnodeGetNumber(node);

  /* operator [] will default construct and insert an object if doesn't exist. */
  int& opt = node_opt_cache_[node_id];

  /* If the node is already in the cache, return the cached value. */
  if (opt) {
    return node_opt_cache_[node_id];
  }

  /* Otherwise, compute the optimality, cache it, and return it. Non-optimal
     solutions have optimality -1. The other solutions have optimality >= 0. */
  opt = -1;

  /* The root is guaranteed to be optimal. */
  if (SCIPnodeGetDepth(node) == 0) {
    opt = solutions_.size() - 1;
    return node_opt_cache_[node_id];
  }

  /* If the node's parent is not optimal, neither is this node. */
  SCIP_NODE* parent = SCIPnodeGetParent(node);
  if (GetOptimality(parent) < 0) {
    return node_opt_cache_[node_id];
  }

  /* Do the full optimality computation. */
  SCIP_VAR** branchvars;      /* vars on which ancestors have branched */
  SCIP_Real* branchbounds;    /* bounds set by ancestor branchings */
  SCIP_BOUNDTYPE* boundtypes; /* bound types set by ancestor branchings */

  /* number of variables on which branchings have been performed in all
   * ancestors; if this is larger than the array size, arrays should be
   * reallocated and the method should be called again */
  int nbranchvars;

  /* available slots in arrays */
  int branchvarssize = 1;

  /* memory allocation and setting nbranchvars */
  SCIP_CALL(SCIPallocBufferArray(scip_, &branchvars, branchvarssize));
  SCIP_CALL(SCIPallocBufferArray(scip_, &branchbounds, branchvarssize));
  SCIP_CALL(SCIPallocBufferArray(scip_, &boundtypes, branchvarssize));
  SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes,
                              &nbranchvars, branchvarssize);

  /* if the arrays were too small, we have to reallocate them and re-call
   * SCIPnodeGetParentBranchings to store all nbranchvars variables */
  if (nbranchvars > branchvarssize) {
    branchvarssize = nbranchvars;
    SCIP_CALL(SCIPreallocBufferArray(scip_, &branchvars, branchvarssize));
    SCIP_CALL(SCIPreallocBufferArray(scip_, &branchbounds, branchvarssize));
    SCIP_CALL(SCIPreallocBufferArray(scip_, &boundtypes, branchvarssize));

    SCIPnodeGetParentBranchings(node, branchvars, branchbounds, boundtypes,
                                &nbranchvars, branchvarssize);
    assert(nbranchvars == branchvarssize);
  }
  assert(nbranchvars >= 1);

  /* Solutions is ordered in increasing order of node optimality. Thus,
   * iterate in reverse order to determine the best solution that is
   * still in the feasible set of this node's subtree. By default,
   * assume the node is not optimal. */
  for (int i = solutions_.size() - 1; i >= 0 && opt < 0; --i) {
    /* A node is optimal wrt a given solution if the solution can be
     * achieved by a descendant of that node. Thus, assume the solution
     * is feasible under this node and then check if we are wrong. */
    bool is_opt = true;
    SCIP_SOL* sol = solutions_[i];
    for (int j = 0; j < nbranchvars && is_opt; ++j) {
      SCIP_Real optsol = SCIPgetSolVal(scip_, sol, branchvars[j]);
      if ((boundtypes[j] == SCIP_BOUNDTYPE_LOWER && optsol < branchbounds[j]) ||
          (boundtypes[j] == SCIP_BOUNDTYPE_UPPER && optsol > branchbounds[j])) {
        is_opt = false;
      }
    }

    /* This gets called only for the highest solution index i (i.e. the
     * best solution) with respect to which our node is optimal, since
     * the for loop breaks when opt >= 0. */
    if (is_opt) opt = i;
  }

  /* free all local memory */
  SCIPfreeBufferArray(scip_, &branchvars);
  SCIPfreeBufferArray(scip_, &boundtypes);
  SCIPfreeBufferArray(scip_, &branchbounds);

  return node_opt_cache_[node_id];
}

/** Get distance from closest optimal node. */
int Oracle::GetDistanceFromOpt(SCIP_NODE* node) {
  NodeId node_id = SCIPnodeGetNumber(node);

  /* If the node is already in the cache, return the cached value. */
  auto it = opt_dist_cache_.find(node_id);
  if (it != opt_dist_cache_.end()) {
    return it->second;
  }
  
  /* Otherwise, compute the distance from the closest optimal node. 
     TODO: what to do in case of multiple optimal nodes? */
  int dist = 0;
  if (GetOptimality(node) >= 0) {
    /* This node is optimal, distance is 0. */
    dist = 0;
  } else {
    /* Node adds 1 to the distance from its parent. */
    SCIP_NODE* parent = SCIPnodeGetParent(node);
    dist = GetDistanceFromOpt(parent) + 1;
  }

  // std::cerr << "Distance: " << dist << "\n";

  /* Cache and return. */
  opt_dist_cache_[node_id] = dist;
  return dist;
}

}  // namespace imilp