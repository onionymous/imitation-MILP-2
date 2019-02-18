/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   eventhdlr_collectdata.cpp
 * @brief  event handler for collecting training data for imitation learning
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "eventhdlr_collectdata.hpp"
#include "objscip/objscip.h"

/**************************************************************************/
/* Overwritten methods of SCIP event handler class */
/**************************************************************************/

/** destructor of event handler to free user data (called when SCIP is exiting)
 */
SCIP_DECL_EVENTFREE(imilp::EventhdlrCollectData::scip_free) {
  return SCIP_OKAY;
} /*lint !e715*/

/** Initialization method of event handler (called after problem was
 *  transformed) */
SCIP_DECL_EVENTINIT(imilp::EventhdlrCollectData::scip_init) {
  imilp::EventhdlrCollectData* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrCollectData*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->Init();

  return SCIP_OKAY;
} /*lint !e715*/

/** Deinitialization method of event handler (called before transformed problem
 * is freed). */
SCIP_DECL_EVENTEXIT(imilp::EventhdlrCollectData::scip_exit) {
  imilp::EventhdlrCollectData* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrCollectData*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->DeInit();

  return SCIP_OKAY;
} /*lint !e715*/

/** Solving process initialization method of event handler (called when branch
 *  and bound process is about to begin).
 *
 *  This method is called when the presolving was finished and the branch and
 *  bound process is about to begin. The event handler may use this call to
 *  initialize its branch and bound specific data.
 *
 */
SCIP_DECL_EVENTINITSOL(imilp::EventhdlrCollectData::scip_initsol) {
  assert(scip != NULL);
  SCIP_CALL(
      SCIPcatchEvent(scip, SCIP_EVENTTYPE_NODESOLVED, eventhdlr, NULL, NULL));
  return SCIP_OKAY;
}

/** Solving process deinitialization method of event handler (called before
 *  branch and bound process data is freed)
 *
 *  This method is called before the branch and bound process is freed.
 *  The event handler should use this call to clean up its branch and bound
 *  data.
 */
SCIP_DECL_EVENTEXITSOL(imilp::EventhdlrCollectData::scip_exitsol) {
  assert(scip != NULL);
  SCIP_CALL(
      SCIPdropEvent(scip, SCIP_EVENTTYPE_NODESOLVED, eventhdlr, NULL, -1));

  return SCIP_OKAY;
}

/** Frees specific constraint data.*/
SCIP_DECL_EVENTDELETE(imilp::EventhdlrCollectData::scip_delete) {
  return SCIP_OKAY;
} /*lint !e715*/

/** execution method of event handler
 *
 *  Processes the event. The method is called every time an event occurs, for
 *  which the event handler is responsible. Event handlers may declare
 * themselves resposible for events by calling the corresponding SCIPcatch...()
 * method. This method creates an event filter object to point to the given
 * event handler and event data.
 */
SCIP_DECL_EVENTEXEC(imilp::EventhdlrCollectData::scip_exec) {
  imilp::EventhdlrCollectData* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrCollectData*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->Process();

  return SCIP_OKAY;
} /*lint !e715*/