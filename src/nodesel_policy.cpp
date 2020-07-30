/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   nodesel_policy.cpp
 * @brief  Custom SCIP node selector that follows a trained imitatoin learning
           policy.
 * @author Stephanie Ding
 */
/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <random>

#include "nodesel_policy.hpp"
#include "objscip/objscip.h"

/**************************************************************************/
/* Overwritten methods of SCIP node selector class */
/**************************************************************************/

/** destructor of node selector to free user data (called when SCIP is exiting)
 *
 *  @see SCIP_DECL_NODESELFREE(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELFREE(imilp::NodeselPolicy::scip_free) {
  return SCIP_OKAY;
}

/** initialization method of node selector (called after problem was transformed)
 *
 *  @see SCIP_DECL_NODESELINIT(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELINIT(imilp::NodeselPolicy::scip_init) {
  imilp::NodeselPolicy* nodesel_obj = NULL;

  assert(scip != NULL);
  nodesel_obj =
      static_cast<imilp::NodeselPolicy*>(SCIPgetObjNodesel(scip, nodesel));
  assert(nodesel_obj != NULL);

  nodesel_obj->Init();

  return SCIP_OKAY;
}

/** deinitialization method of node selector (called before transformed problem is freed)
 *
 *  @see SCIP_DECL_NODESELEXIT(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELEXIT(imilp::NodeselPolicy::scip_exit) {
  imilp::NodeselPolicy* nodesel_obj = NULL;

  assert(scip != NULL);
  nodesel_obj =
      static_cast<imilp::NodeselPolicy*>(SCIPgetObjNodesel(scip, nodesel));
  assert(nodesel_obj != NULL);

  nodesel_obj->DeInit();

  return SCIP_OKAY;
}

/** solving process initialization method of node selector (called when branch and bound process is about to begin)
 *
 *  @see SCIP_DECL_NODESELINITSOL(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELINITSOL(imilp::NodeselPolicy::scip_initsol) {
  return SCIP_OKAY;
}

/** solving process deinitialization method of node selector (called before branch and bound process data is freed)
 *
 *  @see SCIP_DECL_NODESELEXITSOL(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELEXITSOL(imilp::NodeselPolicy::scip_exitsol) {
  return SCIP_OKAY;
}

/** node selection method of node selector
 *
 *  @see SCIP_DECL_NODESELSELECT(x) in @ref type_nodesel.h
 */
SCIP_DECL_NODESELSELECT(imilp::NodeselPolicy::scip_select) {
  assert(nodesel != NULL);
  assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
  assert(scip != NULL);
  assert(selnode != NULL);

  /* Generate a random number. With 5% probability follow the oracle. */
  if (is_train_) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    double prob_oracle = dist(gen);
    if (prob_oracle <= 0.05) {
      bool found = false;

      /* Loop through all nodes in the priority queue. */
      for (int v = 2; v >= 0; --v) {
        SCIP_NODE* node;
        SCIP_NODE** open_nodes;
        int num_open_nodes = -1;

        SCIP_RETCODE retcode;

        switch (v) {
          case 2:
            retcode = SCIPgetChildren(scip_, &open_nodes, &num_open_nodes);
            assert(retcode == SCIP_OKAY);
            break;
          case 1:
            retcode = SCIPgetSiblings(scip_, &open_nodes, &num_open_nodes);
            assert(retcode == SCIP_OKAY);
            break;
          case 0:
            retcode = SCIPgetLeaves(scip_, &open_nodes, &num_open_nodes);
            assert(retcode == SCIP_OKAY);
            break;
          default:
            assert(0);
            break;
        }

        assert(num_open_nodes >= 0);

        /* Loop through each open node and compute features. */
        for (int n = num_open_nodes - 1; n >= 0 && !SCIPisStopped(scip_); --n) {
          node = open_nodes[n];

          if (!SCIPisInfinity(scip_, SCIPnodeGetLowerbound(node))) {
            int opt = oracle_->GetOptimality(node);

            if (opt > -1) {
              /* Node is optimal. */
              *selnode = node;
              found = true;
              break;

            }
          }
        }

        if (found) {
          break;
        }
      }
    } else {
      *selnode = SCIPgetBestNode(scip);
    }
  } else {
    /* If not in training mode, use model only. */
    *selnode = SCIPgetBestNode(scip);
  }
  
  if (dc_) {
    dc_->Process();
  }

  return SCIP_OKAY;
}

/** node comparison method of node selector. defaults to DFS.
 *
 *  @see SCIP_DECL_NODESELCOMP(x) in @ref type_nodesel.h
 *  comparison value for node1 and node2 corresponding to a ranking;
 *         <0 means prefer node1 to node2, >0 means prefer node2 to node1,
 *         and ==0 means no preference
 */
SCIP_DECL_NODESELCOMP(imilp::NodeselPolicy::scip_comp) {
  // auto start = std::chrono::high_resolution_clock::now(); 

  assert(nodesel != NULL);
  assert(strcmp(SCIPnodeselGetName(nodesel), NODESEL_NAME) == 0);
  assert(scip != NULL);

  imilp::NodeselPolicy* nodesel_obj = NULL;

  assert(scip != NULL);
  nodesel_obj =
      static_cast<imilp::NodeselPolicy*>(SCIPgetObjNodesel(scip, nodesel));
  assert(nodesel_obj != NULL);

  // SCIP_Real lowerbound = SCIPgetLowerbound(scip);

  int comp;

  /* Use scorer first. */
  comp = nodesel_obj->Compare(node1, node2);
  if (comp == 0) {

    /* default to DFS */
    int depth1 = SCIPnodeGetDepth(node1);
    int depth2 = SCIPnodeGetDepth(node2);

    if (depth1 > depth2) {
      comp = -1;
    } else if (depth2 > depth1) {
      comp = +1;
    } else {
      /* default to comparing bounds */
      SCIP_Real lowerbound1 = SCIPnodeGetLowerbound(node1);
      SCIP_Real lowerbound2 = SCIPnodeGetLowerbound(node2);

      if (SCIPisLT(scip, lowerbound1, lowerbound2)) {
        comp = -1;
      } else if (SCIPisGT(scip, lowerbound1, lowerbound2)) {
        comp = +1;
      }
    }

    /* if none of these worked. */
    comp = 0;
  }

  // auto stop = std::chrono::high_resolution_clock::now(); 
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 

  // std::cout << "nodesel_policy::scip_comp(): " << duration.count() << "\n";

  return comp;
}