/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   imitation_milp.hpp
 * @brief  Main ImitationMILP solver class.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef IMITATION_MILP_HPP
#define IMITATION_MILP_HPP

#include <string>

#include "eventhdlr_collectdata.hpp"
#include "feat.hpp"
#include "nodesel_policy.hpp"
#include "objscip/objscip.h"
#include "oracle.hpp"
#include "ranknet_model.hpp"
#include "scip/scip.h"

namespace imilp {

/** Base class for the main ImitationMILP instance. */
class ImitationMILP {
 public:
  /** Constructor. */
  ImitationMILP(const std::string& settings_file)
      : settings_file_(settings_file), scip_(NULL) {}

  /** Destructor. */
  ~ImitationMILP() {}

  /** Solve all problems in a directory. */
  bool SolveAll(const std::string& problems_path,
                const std::string& output_path, const std::string& model_file) {
    return true;
  }

  /** Solve a single problem instance. */
  bool Solve(const std::string& problem_file, const std::string& output_file,
             const std::string& model_file);

  /** Begin the training. */
  bool Train(const std::string& train_path, const std::string& valid_path,
             const std::string& model_path, const std::string& prev_model,
             int num_iters, int num_epochs, int batch_size);

  /** Write oracle trajectories to file. */
  bool GetOracleTrajectories(const std::string& problems_path_str,
                             const std::string& data_path_str);

  /** Write trajectories when solving with a model/policy to file. */
  bool GetPolicyTrajectories(const std::string& problems_path_str,
                             const std::string& data_path_str,
                             const std::string& model_path_str);

 private:
  /** Create a new current SCIP instance. */
  SCIP_RETCODE CreateNewSCIP();

  /** Free current SCIP instance. */
  SCIP_RETCODE FreeSCIP();

  /** Call SCIP to solve a problem. Internal. */
  SCIP_RETCODE SolveSCIP(const std::string& problem_name,
                         const std::string& problem_dir,
                         const std::string& output_dir,
                         EventhdlrCollectData* eventhdlr,
                         NodeselPolicy* nodesel, Oracle* oracle);

  /** Helper functions for training loop. */
  bool ValidateDirectoryStructure(const std::string& problems_path);
  bool OracleSolve(const std::string& problems_path,
                   const std::string& data_path, Feat* feat, bool is_append);
  bool PolicySolve(const std::string& problems_path,
                   const std::string& data_path, Feat* feat,
                   RankNetModel* model, double dc_sample_rate, bool is_append);

  /** Path of SCIP params file. */
  std::string settings_file_;

  /** Current SCIP instance. */
  SCIP* scip_;
};

}  // namespace imilp

#endif