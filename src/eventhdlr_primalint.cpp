/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   eventhdlr_primalint.cpp
 * @brief  event handler for calculating primal integral
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#include "eventhdlr_primalint.hpp"
#include "objscip/objscip.h"

namespace imilp {
  void EventhdlrPrimalInt::Init() {
    if (!oracle_->LoadSolutions()) {
      assert(0 && "[FATAL]: EventhdlrPrimalInt: Oracle failed to initialize.");
    }
  }

  void EventhdlrPrimalInt::Process() {
    SCIP_SOL *sol = SCIPgetBestSol(scip_);

    if (sol) {
      double obj = SCIPsolGetOrigObj(sol);

      double t = SCIPgetSolvingTime(scip_);

      // long n = SCIPgetNNodes(scip_);

      obj_vals_.push_back(obj);
      times_.push_back(t);
      // nnodes_.push_back(n);
    }
  }

  void EventhdlrPrimalInt::DeInit() {
    // SCIP_SOL *sol = SCIPgetBestSol(scip_);
    // double opt = SCIPsolGetOrigObj(sol);
    assert(obj_vals_.size() == times_.size());

    double opt = oracle_->GetOptObjectiveValue();

    std::cout << "\nObjective values: (opt is " << opt << ")" << "\n";

    double prev_t = 0.0;
    
    // double prev_n = 0.0;
    double prev_p = 1.0;
    double primal_integral = 0.0;

    std::cout << "inf 1.0 0.0" << "\n";

    for (size_t i = 0; i < obj_vals_.size(); i++) {
      double p =
          fabs(opt - obj_vals_[i]) / std::max(fabs(opt), fabs(obj_vals_[i]));
      primal_integral += (prev_p * (times_[i] - prev_t));
      // primal_integral += (prev_p * ((double)nnodes_[i] - prev_n));

      prev_p = p;
      prev_t = times_[i];
      // prev_n = (double)nnodes_[i];

      // primal_integral += (p * (times_[i] - prev_t));

      std::cout << obj_vals_[i] << " " << p << " " << times_[i] << "\n";
      // std::cout << obj_vals_[i] << " " << p << " " << nnodes_[i] << "\n";
    }

    double t_final = SCIPgetSolvingTime(scip_);
    // long n_final = SCIPgetNNodes(scip_);
    primal_integral += prev_p * (t_final - prev_t);
    // primal_integral += prev_p * ((double)n_final - prev_n);
    // std::cout << opt << " " << 0.0 << " " << t_final << "\n";

    // std::cout << opt << " " << 0.0 << " " << n_final << "\n";
    std::cout << "Primal integral: " << primal_integral << "\n\n";
  }
}

/**************************************************************************/
/* Overwritten methods of SCIP event handler class */
/**************************************************************************/

/** destructor of event handler to free user data (called when SCIP is exiting)
 */
SCIP_DECL_EVENTFREE(imilp::EventhdlrPrimalInt::scip_free) {
  return SCIP_OKAY;
} /*lint !e715*/

/** Initialization method of event handler (called after problem was
 *  transformed) */
SCIP_DECL_EVENTINIT(imilp::EventhdlrPrimalInt::scip_init) {
  imilp::EventhdlrPrimalInt* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrPrimalInt*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->Init();

  return SCIP_OKAY;
} /*lint !e715*/

/** Deinitialization method of event handler (called before transformed problem
 * is freed). */
SCIP_DECL_EVENTEXIT(imilp::EventhdlrPrimalInt::scip_exit) {
  imilp::EventhdlrPrimalInt* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrPrimalInt*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->DeInit();
  oracle_->FreeSolutions();

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
SCIP_DECL_EVENTINITSOL(imilp::EventhdlrPrimalInt::scip_initsol) {
  assert(scip != NULL);
  SCIP_CALL(
      SCIPcatchEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND, eventhdlr, NULL, NULL));
  return SCIP_OKAY;
}

/** Solving process deinitialization method of event handler (called before
 *  branch and bound process data is freed)
 *
 *  This method is called before the branch and bound process is freed.
 *  The event handler should use this call to clean up its branch and bound
 *  data.
 */
SCIP_DECL_EVENTEXITSOL(imilp::EventhdlrPrimalInt::scip_exitsol) {
  assert(scip != NULL);
  SCIP_CALL(
      SCIPdropEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND, eventhdlr, NULL, -1));

  return SCIP_OKAY;
}

/** Frees specific constraint data.*/
SCIP_DECL_EVENTDELETE(imilp::EventhdlrPrimalInt::scip_delete) {
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
SCIP_DECL_EVENTEXEC(imilp::EventhdlrPrimalInt::scip_exec) {
  imilp::EventhdlrPrimalInt* eventhdlr_obj = NULL;

  assert(scip != NULL);
  eventhdlr_obj =
      static_cast<imilp::EventhdlrPrimalInt*>(SCIPgetObjEventhdlr(scip, eventhdlr));
  assert(eventhdlr_obj != NULL);

  eventhdlr_obj->Process();

  return SCIP_OKAY;
} /*lint !e715*/