/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   nodesel_policy.hpp
 * @brief  Custom SCIP node selector that follows a trained imitatoin learning
           policy.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef NODESEL_POLICY_HPP
#define NODESEL_POLICY_HPP

#include "scip/scip.h"
#include "objscip/objscip.h"

#include "scorer_base.hpp"

#define NODESEL_NAME "policy"
#define NODESEL_DESC "node selector that selects nodes according to a policy"
#define NODESEL_PRIORITY 999999

namespace imilp {

/**
 *  A Policy overrides SCIP's node selector class and provides a high-level
 *  interface by which a user can specify a set of criteria to rank nodes in
 *  the SCIP tree.
 */
class NodeselPolicy : public scip::ObjNodesel {
 public:
  NodeselPolicy(SCIP *scip, ScorerBase *scorer)
      : ObjNodesel(scip, NODESEL_NAME, NODESEL_DESC, NODESEL_PRIORITY,
                   NODESEL_PRIORITY),
        scorer_(scorer) {}

  ~NodeselPolicy() {}

  /** Node selector initialization tasks. */
  void Init() { scorer_->Init(); }

  /** Node selector deinitialization tasks. */
  void DeInit() { scorer_->DeInit(); }

  /** Scores a node. */
  int Compare(SCIP_NODE *node1, SCIP_NODE *node2) {
    return scorer_->Compare(node1, node2);
  }

  /**************************************************************************/
  /* Overwritten methods of SCIP node selector class */
  /**************************************************************************/

  /** destructor of node selector to free user data (called when SCIP is exiting)
   *
   *  @see SCIP_DECL_NODESELFREE(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELFREE(scip_free);

  /** initialization method of node selector (called after problem was transformed)
   *
   *  @see SCIP_DECL_NODESELINIT(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELINIT(scip_init);

  /** deinitialization method of node selector (called before transformed problem is freed)
   *
   *  @see SCIP_DECL_NODESELEXIT(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELEXIT(scip_exit);

  /** solving process initialization method of node selector (called when branch and bound process is about to begin)
   *
   *  @see SCIP_DECL_NODESELINITSOL(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELINITSOL(scip_initsol);

  /** solving process deinitialization method of node selector (called before branch and bound process data is freed)
   *
   *  @see SCIP_DECL_NODESELEXITSOL(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELEXITSOL(scip_exitsol);

  /** node selection method of node selector
   *
   *  @see SCIP_DECL_NODESELSELECT(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELSELECT(scip_select);

  /** node comparison method of node selector
   *
   *  @see SCIP_DECL_NODESELCOMP(x) in @ref type_nodesel.h
   */
  virtual SCIP_DECL_NODESELCOMP(scip_comp);

 private:
  /** Scorer that assigns a score to a node. */
  ScorerBase *scorer_;
};

}  // namespace imilp

#endif