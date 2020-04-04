/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   eventhdlr_primalint.hpp
 * @brief  event handler for calculating primal integral
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef EVENTHDLR_PRIMALINT_HPP
#define EVENTHDLR_PRIMALINT_HPP

#include <unordered_map>
#include <string>
#include <vector>

#include "scip/scip.h"
#include "objscip/objscip.h"

#include "data_collector_base.hpp"
#include "oracle.hpp"

namespace imilp {

/** C++ wrapper object for event handlers */
class EventhdlrPrimalInt : public scip::ObjEventhdlr {
 public:
  /** Default constructor. */
  EventhdlrPrimalInt(SCIP *scip, Oracle *oracle)
      : ObjEventhdlr(scip, "primalint",
                     "event handler for calculating primal integral"),
        oracle_(oracle)
       {}

  /** Destructor. */
  ~EventhdlrPrimalInt() {}

  /** Process method for current SCIP state. */
  void Process();

  /** Event handler initialization tasks. */
  void Init() {} ;

  /** Event handler deinitialization tasks. */
  void DeInit();

  /**************************************************************************/
  /* Overwritten methods of SCIP event handler class */
  /**************************************************************************/

  /** Destructor of event handler to free user data (called when SCIP is
   *  exiting). */
  virtual SCIP_DECL_EVENTFREE(scip_free);

  /** Initialization method of event handler (called after problem was
   *  transformed). */
  virtual SCIP_DECL_EVENTINIT(scip_init);

  /** Deinitialization method of event handler (called before transformed
   *  problem is freed). */
  virtual SCIP_DECL_EVENTEXIT(scip_exit);

  /** Solving process initialization method of event handler (called when branch
   *  and bound process is about to begin).
   *
   *  This method is called when the presolving was finished and the branch and
   *  bound process is about to begin. The event handler may use this call to
   *  initialize its branch and bound specific data.
   *
   */
  virtual SCIP_DECL_EVENTINITSOL(scip_initsol);

  /** Solving process deinitialization method of event handler (called before
   *  branch and bound process data is freed)
   *
   *  This method is called before the branch and bound process is freed.
   *  The event handler should use this call to clean up its branch and bound
   *  data.
   */
  virtual SCIP_DECL_EVENTEXITSOL(scip_exitsol);

  /** frees specific constraint data */
  virtual SCIP_DECL_EVENTDELETE(scip_delete);

  /** execution method of event handler
   *
   *  Processes the event. The method is called every time an event occurs, for
   *  which the event handler is responsible. Event handlers may declare
   *  themselves resposible for events by calling the corresponding
   *  SCIPcatch...() method. This method creates an event filter object to point
   *  to the given event handler and event data.
   */
  virtual SCIP_DECL_EVENTEXEC(scip_exec);

 private:
  /** NOTE: Also has an scip_ object, inherited from parent. */

  /** Oracle pointer. Not owned. */
  Oracle *oracle_;

  /** Objective values over time. */
  std::vector<SCIP_Real> obj_vals_;

  /** Time at which a new best primal solution was found. */
  std::vector<SCIP_Real> times_;

  /** Number of nodes when a new best primal solution was found. */
  std::vector<SCIP_Longint> nnodes_;
}; /*lint !e1712*/

}  // namespace imilp

// extern
//  SCIP_RETCODE SCIPincludeObjEventhdlr(
//     SCIP*                 scip,               /**< SCIP data structure */
//     scip::ObjEventhdlr*   objeventhdlr,       /**< event handler object */
//     SCIP_Bool             deleteobject        /**< should the event handler object be deleted when eventhdlristic is freed? */
//     );
 
//  /** returns the eventhdlr object of the given name, or 0 if not existing */
//  extern
//  scip::ObjEventhdlr* SCIPfindObjEventhdlr(
//     SCIP*                 scip,               /**< SCIP data structure */
//     const char*           name                /**< name of event handler */
//     );
 
//  /** returns the eventhdlr object for the given event handler */
//  extern
//  scip::ObjEventhdlr* SCIPgetObjEventhdlr(
//     SCIP*                 scip,               /**< SCIP data structure */
//     SCIP_EVENTHDLR*       eventhdlr           /**< event handler */
//     );


#endif