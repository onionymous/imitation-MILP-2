#include "imitation_milp.hpp"

#include <iostream>

#include "boost/filesystem.hpp"
#include "scip/scipdefplugins.h"

#include "oracle.hpp"
#include "data_collector_base.hpp"
#include "ranked_pairs_collector.hpp"
#include "feat_computer_collector.hpp"
#include "scorer_base.hpp"
#include "oracle_scorer.hpp"
#include "python_scorer.hpp"
#include "ranknet_model.hpp"
#include "feat.hpp"

namespace imilp {

namespace fs = boost::filesystem;

const std::string kSolutionsDirName = "solutions";

/** Solve a problem. */
bool ImitationMILP::Solve(const std::string& problem_file,
                          const std::string& output_file,
                          const std::string& model_file) {
  bool success = true;

  std::cout << "[INFO]: "
            << "ImitationMILP: "
            << "Solving problem: " << problem_file
            << (model_file == "" ? " with default SCIP "
                                 : (" with model: " + model_file)) << "\n";
  
  SCIP_RETCODE retcode;

  retcode = CreateNewSCIP();
  if (retcode != SCIP_OKAY) {
    return false;
  }

  /* Solve current SCIP instance with model. If model file is empty, default
     SCIP will be used to solve. */
  Feat *feat = NULL;
  RankNetModel *model = NULL;

  FeatComputerCollector *dc = NULL;
  EventhdlrCollectData *eventhdlr = NULL;

  PythonScorer *scorer = NULL;
  NodeselPolicy *nodesel = NULL;

  if (model_file != "") {
    feat = new Feat();
  
    /* Create the model. */
    model = new RankNetModel(feat->GetNumFeatures(), model_file, model_file);
    success = model->Init();
    if (!success) {
      return success;
    }

    /* Create the data collector. */
    dc = new FeatComputerCollector(scip_, feat);
    eventhdlr = new EventhdlrCollectData(scip_, dc);

    /* Create the node selector. */
    scorer = new PythonScorer(scip_, model, feat);
    nodesel = new NodeselPolicy(scip_, scorer);

    /* Use eventhandler. */
    SCIP_CALL( SCIPincludeObjEventhdlr(scip_, eventhdlr, FALSE) );

    /* Use node selector. */
    SCIP_CALL( SCIPincludeObjNodesel(scip_, nodesel, FALSE) );
  }

  /* Read settings. */
  SCIP_CALL( SCIPreadParams(scip_, settings_file_.c_str()) );
  
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

  std::cerr << "[INFO]: " << "Read problem from file: " << problem_file << "\n";

  /*******************
   * Problem Solving *
   *******************/

  /* solve problem */
  std::cerr << "[INFO]: " << "Solving problem..." << "\n\n";

 /***********************************
  * Version and library information *
  ***********************************/
  SCIP_CALL(SCIPsolve(scip_));

  std::cerr << "\n";

  /* If an output file was specified, write the solution to it. */
  if (output_file != "") {
    std::cerr << "[INFO]: " << "ImitationMILP: " << "Writing solution to: " << output_file << "\n";
    FILE* file = fopen(output_file.c_str(), "w");
    SCIP_CALL(SCIPprintBestSol(scip_, file, FALSE));
    fclose(file);
  }

  /* Clean up current SCIP instance. */
  retcode = FreeSCIP();

  if (retcode != SCIP_OKAY) {
    success = false;
  }

  delete feat;
  delete model;
  delete dc;
  delete eventhdlr;
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

  /* Make sure the main solutions directory exists. */
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
                          NULL /* use node selector */);
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
bool ImitationMILP::OracleSolve(const std::string& problems_path, Feat* feat) {
  bool success = true;
  SCIP_RETCODE retcode;

  fs::path path = fs::path(problems_path);
  fs::path solutions_dir = path / kSolutionsDirName;
  fs::path data_file = path / (problems_path + ".data");

  for (auto& problem : fs::directory_iterator(path)) {
      /* not a valid problem file, skip. */
      if (fs::extension(problem.path()) != ".lp") {
        continue;
      }

      fs::path problem_name = problem.path().stem();
      fs::path problem_solutions_dir = fs::path(solutions_dir) / problem_name;

      retcode = CreateNewSCIP();
      if (retcode != SCIP_OKAY) {
        success = false;
        break;
      }

      /* Create the oracle. */
      Oracle oracle(scip_, problem_solutions_dir.string());

      /* Create the data collector. */
      RankedPairsCollector dc(scip_, data_file.string(), &oracle, feat,
                              1.0 /* data collector random sampling rate */);
      EventhdlrCollectData event_handler(scip_, &dc);

      /* Create the oracle node selector. */
      OracleScorer scorer(scip_, &oracle);
      NodeselPolicy node_selector(scip_, &scorer);

      /* Solve current SCIP instance. */
      retcode = SolveSCIP(problem_name.string() /* problem name */,
                          problems_path /* problem input file */,
                          std::string("") /* don't write solution to outfile */,
                          &event_handler, &node_selector);
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
bool ImitationMILP::PolicySolve(const std::string& problems_path, Feat* feat,
                                RankNetModel* model, double dc_sample_rate) {
  bool success = true;
  SCIP_RETCODE retcode;

  fs::path path = fs::path(problems_path);
  fs::path solutions_dir = path / kSolutionsDirName;
  fs::path data_file = path / (problems_path + ".data");

  for (auto& problem : fs::directory_iterator(path)) {
    /* not a valid problem file, skip. */
    if (fs::extension(problem.path()) != ".lp") {
      continue;
    }

    fs::path problem_name = problem.path().stem();
    fs::path problem_solutions_dir = fs::path(solutions_dir) / problem_name;

    retcode = CreateNewSCIP();
    if (retcode != SCIP_OKAY) {
      success = false;
      break;
    }

    /* Create the oracle. */
    Oracle oracle(scip_, problem_solutions_dir.string());

    /* Create the data collector. */
    RankedPairsCollector dc(scip_, data_file.string(), &oracle, feat, 0.2);
    EventhdlrCollectData event_handler(scip_, &dc);

    /* Create the node selector. */
    PythonScorer scorer(scip_, model, feat);
    NodeselPolicy node_selector(scip_, &scorer);

    /* Solve current SCIP instance. */
    retcode = SolveSCIP(problem_name.string() /* problem name */,
                        problems_path /* problem input file */,
                        std::string("") /* don't write solution to outfile */,
                        &event_handler, &node_selector);
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
bool ImitationMILP::Train(const std::string& train_path,
                          const std::string& valid_path,
                          const std::string& model_path,
                          const std::string& prev_model, int num_iters,
                          int num_epochs, int batch_size) {
  bool success = true;

  /* Validate train and validation directory structures and make sure the 
     problems have valid solutions. */
  if (!ValidateDirectoryStructure(train_path) ||
      !ValidateDirectoryStructure(valid_path)) {
    return false;
  }

  /* Setup training. */
  /* Features to compute during training. */
  Feat feat;
  
  /* Create the model. */
  RankNetModel model(feat.GetNumFeatures(), model_path, prev_model);
  success = model.Init();
  if (!success) {
    return success;
  }

  /* If there was no previous model, train an initial model with the oracle
     scorer as behavioral cloning. */
  if (prev_model == "") {
    std::cerr << "[INFO]: "
              << "ImitationMILP: "
              << "Previous model was not specified, training a new model and "
                 "saving to file: " << model_path << "\n";

    /* Collect initial data using oracle policy. */
    if (!OracleSolve(train_path, &feat) || 
        !OracleSolve(valid_path, &feat)) {
      return false;
    }

    /* Train the initial model. */
    fs::path train_file = fs::path(train_path) / (train_path + ".data");
    fs::path valid_file = fs::path(valid_path) / (valid_path + ".data");
    success = model.Train(train_file.string(), valid_file.string(), num_epochs,
                          batch_size);
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
    std::cerr << "[INFO]: " << "Starting train iteration " << i << "\n";

    /* Collect data using current trained model. */
    if (!PolicySolve(train_path, &feat, &model, 0.05) ||
        !PolicySolve(valid_path, &feat, &model, 1.0)) {
      return false;
    }

    /* Train the next model. */
    fs::path train_file = fs::path(train_path) / (train_path + ".data");
    fs::path valid_file = fs::path(valid_path) / (valid_path + ".data");
    success = model.Train(train_file.string(), valid_file.string(), num_epochs,
                batch_size);
  }

  return success;
}

/** Create a new current SCIP instance. */
SCIP_RETCODE ImitationMILP::CreateNewSCIP() {
  assert(scip_ == NULL);

  SCIP_CALL( SCIPcreate(&scip_) );
  SCIP_CALL( SCIPincludeDefaultPlugins(scip_) );

  return SCIP_OKAY;
}

/** Free the current SCIP instance. */
SCIP_RETCODE ImitationMILP::FreeSCIP() {
  if (scip_ != NULL) {
    SCIP_CALL( SCIPfree(&scip_) );
    BMScheckEmptyMemory();
    scip_ = NULL;
  }

  return SCIP_OKAY;
}


/** Base class for the main ImitationMILP instance. */
SCIP_RETCODE ImitationMILP::SolveSCIP(const std::string& problem_name,
                                      const std::string& problem_dir,
                                      const std::string& output_dir,
                                      EventhdlrCollectData *eventhdlr,
                                      NodeselPolicy *nodesel) {
  /* Use eventhandler if specified. */
  if (eventhdlr) {
    SCIP_CALL( SCIPincludeObjEventhdlr(scip_, eventhdlr, FALSE) );
  }

  /* Use node selector if specified. */
  if (nodesel) {
    SCIP_CALL( SCIPincludeObjNodesel(scip_, nodesel, FALSE) );
  }

  /* Read settings. */
  SCIP_CALL( SCIPreadParams(scip_, settings_file_.c_str()) );
  
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

  std::cerr << "[INFO]: " << "Read problem from file: " << problem_file << "\n";

  /*******************
   * Problem Solving *
   *******************/

  /* solve problem */
  std::cerr << "[INFO]: " << "Solving problem..." << "\n\n";

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
    std::cerr << "[INFO]: " << "ImitationMILP: " << "Writing solution to: " << output_file << "\n";
    FILE* file = fopen(output_file.string().c_str(), "w");
    SCIP_CALL(SCIPprintBestSol(scip_, file, FALSE));
    fclose(file);
  }

  // SCIP_CALL( SCIPprintStatistics(scip_, NULL) );

  // Calculate some problem-solving stats
  // SCIP_Real duality_gap = SCIPgetGap(scip_);
  // SCIP_Real best = SCIPgetBestSol(scip_)->obj;
  // SCIP_Real opt = std::numeric_limits<double>::infinity();
  // SCIP_Real opt_gap = abs(opt - best) / std::min(abs(opt), abs(best));
  // SCIP_Longint nsols = SCIPgetNSolsFound(scip_);
  // SCIP_Longint nBestSols = SCIPgetNBestSolsFound(scip_);
  // SCIP_Longint nnodes = SCIPgetNNodes(scip_);

  // Print those stats out
  // std::ofstream stats;
  // stats.open(std::string(results_name + ".stats").c_str(),
  //            std::ofstream::out | std::ofstream::app);

  // std::cerr << duality_gap << ",";
  // std::cerr << opt_gap << ",";
  // std::cerr << nsols << ",";
  // std::cerr << nBestSols << ",";
  // std::cerr << nnodes << "\n";
  // stats.close();

  return SCIP_OKAY;
}

}  // namespace imilp