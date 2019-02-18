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

  *selnode = SCIPgetBestNode(scip);

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

  return comp;
}