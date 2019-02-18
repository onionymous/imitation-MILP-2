/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   scorer_base.hpp
 * @brief  Base class for an object that scores a node.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef SCORER_BASE_HPP
#define SCORER_BASE_HPP

#include "scip/scip.h"

namespace imilp {

/** Base class for a data collector class that performs some processing on the
    current nodes, given an SCIP instance. */
class ScorerBase {
 public:
  /** Constructor. */
  ScorerBase(SCIP *scip) : scip_(scip) {};

  /** Initialization tasks. */
  virtual void Init() = 0;

  /** Deinitialization tasks. */
  virtual void DeInit() = 0;

  /** How to process the current SCIP instance. */
  virtual int Compare(SCIP_NODE *node1, SCIP_NODE *node2) = 0;

  /** Virtual destructor. */
  virtual ~ScorerBase() {};

 protected:
  /** Reference to SCIP instance. Not owned. */
  SCIP *scip_;
};

}  // namespace imilp

#endif