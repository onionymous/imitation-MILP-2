/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   imitation_milp.cpp
 * @brief  Main ImitationMILP solver class.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "imitation_milp.hpp"

#include <iostream>

#include "boost/filesystem.hpp"
#include "data_collector_base.hpp"
#include "eventhdlr_primalint.hpp"
#include "feat.hpp"
#include "feat_computer_collector.hpp"
#include "oracle.hpp"
#include "oracle_scorer.hpp"
#include "python_scorer.hpp"
#include "ranked_pairs_collector.hpp"
#include "ranknet_model.hpp"
#include "scip/scipdefplugins.h"
#include "scorer_base.hpp"

namespace imilp {

namespace fs = boost::filesystem;

const std::string kSolutionsDirName = "solutions";
const std::string kDataDirName = "data";

/** Solve a problem. */
bool ImitationMILP::Solve(const std::string& problem_file,
                          const std::string& output_file,
                          const std::string& model_file) {
  bool success = true;

  std::cout << "[INFO]: "
            << "ImitationMILP: "
            << "Solving problem: " << problem_file
            << (model_file == "" ? " with default SCIP "
                                 : (" with model: " + model_file))
            << "\n";

  SCIP_RETCODE retcode;

  retcode = CreateNewSCIP();
  if (retcode != SCIP_OKAY) {
    return false;
  }

  /* Solve current SCIP instance with model. If model file is empty, default
     SCIP will be used to solve. */
  Feat* feat = NULL;
  RankNetModel* model = NULL;

  FeatComputerCollector* dc = NULL;
  EventhdlrCollectData* eventhdlr_dc = NULL;
  EventhdlrPrimalInt* eventhdlr_primalint = NULL;

  PythonScorer* scorer = NULL;
  NodeselPolicy* nodesel = NULL;

  // fix this later ur a dummy
  if (model_file != "") {
    feat = new Feat();

    /* Create the model. */
    model = new RankNetModel(feat->GetNumFeatures(), model_file, model_file);
    success = model->Init(false /* is_gpu */);
    if (!success) {
      return success;
    }

    /* Create the node selector. */
    scorer = new PythonScorer(scip_, model, feat);
    nodesel = new NodeselPolicy(scip_, scorer, NULL);

    /* Create the data collector. */
    dc = new FeatComputerCollector(scip_, feat);
    eventhdlr_dc = new EventhdlrCollectData(scip_, dc);

    /* Use node selector. */
    SCIP_CALL(SCIPincludeObjEventhdlr(scip_, eventhdlr_dc, FALSE));
    SCIP_CALL(SCIPincludeObjNodesel(scip_, nodesel, FALSE));
  }

  /* Use eventhandler for primal integral. */
  eventhdlr_primalint = new EventhdlrPrimalInt(scip_, NULL);
  SCIP_CALL(SCIPincludeObjEventhdlr(scip_, eventhdlr_primalint, FALSE));

  /* Read settings. */
  SCIP_CALL(SCIPreadParams(scip_, settings_file_.c_str()));

  /* Read problem file. */
  retcode = SCIPreadProb(scip_, problem_file.c_str(), NULL);

  switch (retcode) {
    case SCIP_NOFILE:
      SCIPinfoMessage(scip_, NULL, "file <%s> not found\n", problem_file);
      success = false;
      break;
    case SCIP_PLUGINNOTFOUND:
      SCIPinfoMessage(scip_, NULL, "no reader for input file <%s> available\n",
                      problem_file);
      success = false;
      break;
    case SCIP_READERROR:
      SCIPinfoMessage(scip_, NULL, "error reading file <%s>\n", problem_file);
      success = false;
      break;
    default:
      SCIP_CALL(retcode);
  } /*lint !e788*/

  if (!success) {
    FreeSCIP();
    return success;
  }

  std::cerr << "[INFO]: "
            << "Read problem from file: " << problem_file << "\n";

  /*******************
   * Problem Solving *
   *******************/

  /* solve problem */
  std::cerr << "[INFO]: "
            << "Solving problem..."
            << "\n\n";

  /***********************************
   * Version and library information *
   ***********************************/
  SCIP_CALL(SCIPsolve(scip_));

  std::cerr << "\n";

  /* If an output file was specified, write the solution to it. */
  if (output_file != "") {
    std::cerr << "[INFO]: "
              << "ImitationMILP: "
              << "Writing solution to: " << output_file << "\n";
    FILE* file = fopen(output_file.c_str(), "w");
    SCIP_CALL(SCIPprintBestSol(scip_, file, FALSE));
    fclose(file);
  }

  SCIPprintStatusStatistics(scip_, NULL);
  SCIPprintOrigProblemStatistics(scip_, NULL);
  SCIPprintTimingStatistics(scip_, NULL);
  SCIPprintTreeStatistics(scip_, NULL);
  SCIPprintLPStatistics(scip_, NULL);
  SCIPprintSolutionStatistics(scip_, NULL);

  /* Clean up current SCIP instance. */
  retcode = FreeSCIP();

  if (retcode != SCIP_OKAY) {
    success = false;
  }

  delete feat;
  delete model;
  delete dc;
  delete eventhdlr_dc;
  delete eventhdlr_primalint;
  delete scorer;
  delete nodesel;

  return success;
}

/** Validate directory structure. */
bool ImitationMILP::ValidateDirectoryStructure(
    const std::string& problems_path) {
  bool success = true;
  SCIP_RETCODE retcode;

  /* Iterate through each problem in the problems directory and make sure a
     valid solution structure exists for each problem. */
  std::cout << "[INFO]: "
            << "ImitationMILP: "
            << "Validating directory structure for folder: " << problems_path
            << "\n";

  fs::path path = fs::path(problems_path);
  fs::path solutions_dir = path / kSolutionsDirName;
  fs::path data_dir = path / kDataDirName;

  /* Remove existing data if it exists. */
  if (fs::exists(data_dir) && fs::is_directory(data_dir)) {
    fs::remove_all(data_dir);
  }

  /* Create data directory, where each problem will save its data to a separate
   * file in this directory. */
  if (fs::create_directory(data_dir)) {
    std::cout << "[INFO]: "
              << "ImitationMILP: "
              << "Creating data directory: " << data_dir << "\n";
  }

  /* Make sure the main data and solutions directories exists. */
  fs::create_directory(data_dir);
  fs::create_directory(solutions_dir);

  for (auto& problem : fs::directory_iterator(path)) {
    /* not a valid problem file, skip. */
    if (fs::extension(problem.path()) != ".lp") {
      continue;
    }

    fs::path problem_name = problem.path().stem();
    fs::path problem_solutions_dir = solutions_dir / problem_name;

    /* If the solution directory did not exist, we create it and generate a
     * solution. */
    if (fs::create_directory(problem_solutions_dir)) {
      std::cout
          << "[INFO]: "
          << "ImitationMILP: "
          << "Solutions folder for problem " << problem_name
          << " was not found in specified directory. Generating a solution."
          << "\n";
      /* Create new SCIP instance. */
      retcode = CreateNewSCIP();
      if (retcode != SCIP_OKAY) {
        success = false;
        break;
      }

      /* Solve current SCIP instance. */
      retcode = SolveSCIP(problem_name.string() /* problem name */,
                          problems_path /* input file */,
                          problem_solutions_dir.string() /* output solution */,
                          NULL /* use data collector */,
                          NULL /* use node selector */, NULL);
      if (retcode != SCIP_OKAY) {
        success = false;
        break;
      }

      /* Clean up current SCIP instance. */
      retcode = FreeSCIP();
    }
  }

  /* If something failed. */
  if (!success) {
    FreeSCIP();
    return success;
  }

  return success;
}

/** Solve a directory of problems with the oracle node selector and collect
    data about the nodes. */
bool ImitationMILP::OracleSolve(const std::string& problems_path,
                                const std::string& data_path, Feat* feat,
                                bool is_append) {
  bool success = true;
  SCIP_RETCODE retcode;

  fs::path path = fs::path(problems_path);
  fs::path solutions_dir = path / kSolutionsDirName;
  fs::path data_dir = fs::path(data_path);

  for (auto& problem : fs::directory_iterator(path)) {
    /* not a valid problem file, skip. */
    if (fs::extension(problem.path()) != ".lp") {
      continue;
    }

    fs::path problem_name = problem.path().stem();
    fs::path problem_solutions_dir = fs::path(solutions_dir) / problem_name;
    fs::path data_file = data_dir / (problem_name.string() + ".data");

    retcode = CreateNewSCIP();
    if (retcode != SCIP_OKAY) {
      success = false;
      break;
    }

    /* Create the oracle. */
    Oracle oracle(scip_, problem_solutions_dir.string());

    /* Create the data collector. */
    FeatComputerCollector feat_dc(scip_, feat);
    EventhdlrCollectData event_handler(scip_, &feat_dc);

    /* Create the oracle node selector. */
    OracleScorer scorer(scip_, &oracle);
    RankedPairsCollector dc(
        scip_, data_file.string(), /* file to collect data */
        is_append /* append to existing file */, &oracle, feat,
        1.0 /* data collector random sampling rate */);
    NodeselPolicy node_selector(scip_, &scorer, &dc);

    /* Solve current SCIP instance. */
    retcode = SolveSCIP(problem_name.string() /* problem name */,
                        problems_path /* problem input file */,
                        std::string("") /* don't write solution to outfile */,
                        &event_handler, &node_selector, &oracle);
    if (retcode != SCIP_OKAY) {
      success = false;
      break;
    }

    /* Clean up current SCIP instance. */
    retcode = FreeSCIP();
  }

  /* If something went wrong, free current SCIP instance. */
  if (!success) {
    FreeSCIP();
    return success;
  }

  return success;
}

/** Solve a directory of problems with the policy node selector and collect
    data about the nodes. */
bool ImitationMILP::PolicySolve(const std::string& problems_path,
                                const std::string& data_path, Feat* feat,
                                RankNetModel* model, double dc_sample_rate,
                                bool is_append) {
  bool success = true;
  SCIP_RETCODE retcode;

  fs::path path = fs::path(problems_path);
  fs::path solutions_dir = path / kSolutionsDirName;
  fs::path data_dir = fs::path(data_path);

  for (auto& problem : fs::directory_iterator(path)) {
    /* not a valid problem file, skip. */
    if (fs::extension(problem.path()) != ".lp") {
      continue;
    }

    fs::path problem_name = problem.path().stem();
    fs::path problem_solutions_dir = fs::path(solutions_dir) / problem_name;
    fs::path data_file = data_dir / (problem_name.string() + ".data");

    retcode = CreateNewSCIP();
    if (retcode != SCIP_OKAY) {
      success = false;
      break;
    }

    /* Create the oracle. */
    Oracle oracle(scip_, problem_solutions_dir.string());

    /* Create the data collector. */
    FeatComputerCollector feat_dc(scip_, feat);
    EventhdlrCollectData event_handler(scip_, &feat_dc);

    /* Create the node selector. */
    PythonScorer scorer(scip_, model, feat);
    RankedPairsCollector dc(scip_, data_file.string(), is_append, &oracle, feat,
                            dc_sample_rate);
    NodeselPolicy node_selector(scip_, &scorer, &dc);

    /* Solve current SCIP instance. */
    retcode = SolveSCIP(problem_name.string() /* problem name */,
                        problems_path /* problem input file */,
                        std::string("") /* don't write solution to outfile */,
                        &event_handler, &node_selector, &oracle);
    if (retcode != SCIP_OKAY) {
      success = false;
      break;
    }

    /* Clean up current SCIP instance. */
    retcode = FreeSCIP();

    /* If something went wrong, free current SCIP instance. */
    if (!success) {
      FreeSCIP();
      return success;
    }
  }

  return success;
}

/** Train loop for model. */
bool ImitationMILP::Train(const std::string& train_path_str,
                          const std::string& valid_path_str,
                          const std::string& model_path,
                          const std::string& prev_model, int num_iters,
                          int num_epochs, int batch_size) {
  bool success = true;

  /* Validate train and validation directory structures and make sure the
     problems have valid solutions. */
  if (!ValidateDirectoryStructure(train_path_str) ||
      !ValidateDirectoryStructure(valid_path_str)) {
    return false;
  }

  fs::path train_data_path = fs::path(train_path_str) / kDataDirName;
  fs::path valid_data_path = fs::path(valid_path_str) / kDataDirName;

  /* Setup training. */
  /* Features to compute during training. */
  Feat feat;

  /* Create the model. */
  RankNetModel model(feat.GetNumFeatures(), model_path, prev_model);
  success = model.Init(true /* is_gpu */);
  if (!success) {
    return success;
  }

  /* If there was no previous model, train an initial model with the oracle
     scorer as behavioral cloning. */
  if (prev_model == "") {
    std::cerr << "[INFO]: "
              << "ImitationMILP: "
              << "Previous model was not specified, training a new model and "
                 "saving to file: "
              << model_path << "\n";

    /* Remove existing data if it exists. */

    // if (fs::exists(data_dir) &&  fs::is_directory(data_dir)) {
    //   fs::remove_all(data_dir);
    // }
    // fs::create_directory(data_dir);

    /* Collect initial data using oracle policy. */
    if (!OracleSolve(train_path_str, train_data_path.string(), &feat,
                     true /* aggregate data */) ||
        !OracleSolve(valid_path_str, valid_data_path.string(), &feat, false)) {
      return false;
    }

    /* Train the initial model. */

    success = model.Train(train_data_path.string(), valid_data_path.string(),
                          num_epochs, batch_size);

    if (!success) {
      return success;
    }
  }

  /* Run the train loop for any number of iterations. */
  std::cerr << "[INFO]: "
            << "ImitationMILP: "
            << "Running training loop for " << num_iters << " iterations."
            << "\n";
  for (int i = 0; (i < num_iters) && success; ++i) {
    std::cerr << "[INFO]: "
              << "Starting train iteration " << i << "\n";

    /* Collect data using current trained model. */
    if (!PolicySolve(train_path_str, train_data_path.string(), &feat, &model,
                     1.0 /* sampling rate */, true /* aggregate data */) ||
        !PolicySolve(valid_path_str, valid_data_path.string(), &feat, &model,
                     1.0, false)) {
      return false;
    }

    /* Train the next model. */
    fs::path train_path = fs::path(train_path_str) / kDataDirName;
    fs::path valid_path = fs::path(valid_path_str) / kDataDirName;
    success = model.Train(train_path.string(), valid_path.string(), num_epochs,
                          batch_size);
  }

  return success;
}

/** Write oracle trajectories. */
bool ImitationMILP::GetOracleTrajectories(const std::string& problems_path_str,
                                          const std::string& data_path_str) {
  bool success = true;

  /* Validate train and validation directory structures and make sure the
     problems have valid solutions. */
  if (!ValidateDirectoryStructure(problems_path_str)) {
    return false;
  }

  /* Remove existing data if it exists. */
  fs::path data_dir = fs::path(data_path_str);

  if (fs::exists(data_dir) && fs::is_directory(data_dir)) {
    // fs::remove_all(data_dir);
    std::cerr << "[ERROR]: "
              << "ImitationMILP: "
              << "Data output directory: " << data_dir << "not empty." << "\n";
    return false;
  }

  /* Create data directory, where each problem will save its data to a separate
   * file in this directory. */
  if (fs::create_directory(data_dir)) {
    std::cout << "[INFO]: "
              << "ImitationMILP: "
              << "Writing trajectories to directory: " << data_dir << "\n";
  }

  /* Setup solve. */
  /* Features to compute during solve. */
  Feat feat;

  /* Collect initial data using oracle policy. */
  if (!OracleSolve(problems_path_str, data_path_str, &feat, false)) {
    return false;
  }

  return success;
}

/** Write trajectories from solving with a particular model/policy. */
bool ImitationMILP::GetPolicyTrajectories(const std::string& problems_path_str,
                                          const std::string& data_path_str,
                                          const std::string& model_path_str) {
  bool success = true;

  /* Validate train and validation directory structures and make sure the
     problems have valid solutions. */
  if (!ValidateDirectoryStructure(problems_path_str)) {
    return false;
  }

  /* Remove existing data if it exists. */
  fs::path data_dir = fs::path(data_path_str);

  if (fs::exists(data_dir) && fs::is_directory(data_dir)) {
    // fs::remove_all(data_dir);
    std::cerr << "[ERROR]: "
              << "ImitationMILP: "
              << "Data output directory: " << data_dir << " not empty." << "\n";
    return false;
  }

  /* Create data directory, where each problem will save its data to a separate
   * file in this directory. */
  if (fs::create_directory(data_dir)) {
    std::cout << "[INFO]: "
              << "ImitationMILP: "
              << "Writing trajectories to directory: " << data_dir << "\n";
  }

  /* Check model exists. */
  fs::path model_path = fs::path(model_path_str);
  if (!fs::exists(model_path) || fs::is_directory(model_path)) {
    // fs::remove_all(model_path);
    std::cerr << "[ERROR]: "
              << "ImitationMILP: "
              << "Model file: " << model_path
              << " does not exist or is not a file."
              << "\n";
    return false;
  }

  /* Setup solve. */
  /* Features to compute during solve. */
  Feat feat;

  /* Create the model. */
  RankNetModel model(feat.GetNumFeatures(), model_path_str, model_path_str);
  success = model.Init(true /* is_gpu */);
  if (!success) {
    return success;
  }

  /* Collect initial data using oracle policy. */
  if (!PolicySolve(problems_path_str, data_path_str, &feat, &model,
                   1.0 /* sampling rate */, false /* aggregate data */)) {
    return false;
  }

  return success;
}

/** Create a new current SCIP instance. */
SCIP_RETCODE ImitationMILP::CreateNewSCIP() {
  assert(scip_ == NULL);

  SCIP_CALL(SCIPcreate(&scip_));
  SCIP_CALL(SCIPincludeDefaultPlugins(scip_));

  return SCIP_OKAY;
}

/** Free the current SCIP instance. */
SCIP_RETCODE ImitationMILP::FreeSCIP() {
  if (scip_ != NULL) {
    SCIP_CALL(SCIPfree(&scip_));
    BMScheckEmptyMemory();
    scip_ = NULL;
  }

  return SCIP_OKAY;
}

/** Base class for the main ImitationMILP instance. */
SCIP_RETCODE ImitationMILP::SolveSCIP(const std::string& problem_name,
                                      const std::string& problem_dir,
                                      const std::string& output_dir,
                                      EventhdlrCollectData* eventhdlr,
                                      NodeselPolicy* nodesel, Oracle* oracle) {
  /* Use eventhandler if specified. */
  if (eventhdlr) {
    SCIP_CALL(SCIPincludeObjEventhdlr(scip_, eventhdlr, FALSE));
  }

  /* Use node selector if specified. */
  if (nodesel) {
    SCIP_CALL(SCIPincludeObjNodesel(scip_, nodesel, FALSE));
  }

  if (oracle) {
    EventhdlrPrimalInt* eventhdlr_primalint =
        new EventhdlrPrimalInt(scip_, oracle);
    SCIP_CALL(SCIPincludeObjEventhdlr(scip_, eventhdlr_primalint, TRUE));
  }

  /* Read settings. */
  SCIP_CALL(SCIPreadParams(scip_, settings_file_.c_str()));

  /* Read problem file. */
  SCIP_RETCODE retcode;
  fs::path problem_file = fs::path(problem_dir) / (problem_name + ".lp");
  retcode = SCIPreadProb(scip_, problem_file.string().c_str(), NULL);

  switch (retcode) {
    case SCIP_NOFILE:
      SCIPinfoMessage(scip_, NULL, "file <%s> not found\n", problem_file);
      return SCIP_OKAY;
    case SCIP_PLUGINNOTFOUND:
      SCIPinfoMessage(scip_, NULL, "no reader for input file <%s> available\n",
                      problem_file);
      return SCIP_OKAY;
    case SCIP_READERROR:
      SCIPinfoMessage(scip_, NULL, "error reading file <%s>\n", problem_file);
      return SCIP_OKAY;
    default:
      SCIP_CALL(retcode);
  } /*lint !e788*/

  std::cerr << "[INFO]: "
            << "Read problem from file: " << problem_file << "\n";

  /*******************
   * Problem Solving *
   *******************/

  /* solve problem */
  std::cerr << "[INFO]: "
            << "Solving problem..."
            << "\n\n";

  /***********************************
   * Version and library information *
   ***********************************/
  // SCIPprintVersion(scip_, NULL);
  // SCIPinfoMessage(scip_, NULL, "\n");
  // SCIPprintExternalCodes(scip_, NULL);
  // SCIPinfoMessage(scip_, NULL, "\n");

  SCIP_CALL(SCIPsolve(scip_));

  std::cerr << "\n";

  /* If an output file was specified, write the solution to it. */
  if (output_dir != "") {
    fs::path output_file = fs::path(output_dir) / (problem_name + ".sol");
    std::cerr << "[INFO]: "
              << "ImitationMILP: "
              << "Writing solution to: " << output_file << "\n";
    FILE* file = fopen(output_file.string().c_str(), "w");
    SCIP_CALL(SCIPprintBestSol(scip_, file, FALSE));
    fclose(file);
  }

  // Print out solving statistics
  SCIPprintStatusStatistics(scip_, NULL);
  SCIPprintOrigProblemStatistics(scip_, NULL);
  SCIPprintTimingStatistics(scip_, NULL);
  SCIPprintTreeStatistics(scip_, NULL);
  SCIPprintLPStatistics(scip_, NULL);
  SCIPprintSolutionStatistics(scip_, NULL);

  // SCIP_CALL( SCIPprintStatistics(scip_, NULL) );

  // // Calculate some problem-solving stats
  // SCIP_Real solving_time = SCIPgetSolvingTime(scip_);
  // SCIP_Real duality_gap = SCIPgetGap(scip_);
  // SCIP_Real best = SCIPgetBestSol(scip_)->obj;
  // SCIP_Real opt = std::numeric_limits<double>::infinity();
  // SCIP_Real opt_gap = abs(opt - best) / std::min(abs(opt), abs(best));
  // SCIP_Longint n_sols = SCIPgetNSolsFound(scip_);
  // SCIP_Longint n_best_sols = SCIPgetNBestSolsFound(scip_);
  // SCIP_Longint n_nodes = SCIPgetNNodes(scip_);

  // // Print those stats out
  // // std::ofstream stats;
  // // stats.open(std::string(results_name + ".stats").c_str(),
  // //            std::ofstream::out | std::ofstream::app);

  // std::cout << "Solving stats for problem: " << problem_file << "\n";
  // std::cout << "Solving time (sec): " << solving_time << "\n";
  // std::cout << "Solving nodes: " << n_nodes << "\n";
  // std::cout << "Duality gap: " << duality_gap << "\n";
  // std::cout << "Optimality gap: " << opt_gap << "\n";
  // std::cout << "Solutions: " << n_sols << "\n";
  // std::cout << "Best solutions: " << n_best_sols << "\n";

  return SCIP_OKAY;
}

}  // namespace imilp