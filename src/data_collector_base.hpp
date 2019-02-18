/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   data_collector_base.hpp
 * @brief  Base class for an object that computes features of a node.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef DATA_COLLECTOR_BASE_HPP
#define DATA_COLLECTOR_BASE_HPP

#include "scip/scip.h"

namespace imilp {

/** Base class for a data collector class that performs some processing on the
    current nodes, given an SCIP instance. */
class DataCollectorBase {
 public:
  /** Constructor. */
  DataCollectorBase(SCIP *scip) : scip_(scip) {};

  /** Initialization tasks. */
  virtual void Init() = 0;

  /** Deinitialization tasks. */
  virtual void DeInit() = 0;

  /** How to process the current SCIP instance. */
  virtual void Process() = 0;

  /** Virtual destructor. */
  virtual ~DataCollectorBase() {};

 protected:
  /** Reference to SCIP instance. Not owned. */
  SCIP *scip_;
};

}  // namespace imilp

#endif