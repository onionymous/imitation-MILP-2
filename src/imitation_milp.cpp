#include "imitation_milp.hpp"

#include <iostream>

#include "boost/filesystem.hpp"
#include "scip/scipdefplugins.h"

#include "oracle.hpp"
#include "data_collector_base.hpp"
#include "ranked_pairs_collector.hpp"
#include "scorer_base.hpp"
#include "oracle_scorer.hpp"
#include "python_scorer.hpp"
#include "ranknet_model.hpp"
#include "feat.hpp"

namespace imilp {

namespace fs = boost::filesystem;

/** Solve a thing */
bool ImitationMILP::Train(const std::string& problems_path,
                          const std::string& solutions_path,
                          const std::string& model_path,
                          const std::string& prev_model, int num_iters,
                          int num_epochs, int batch_size) {
  bool success = true;

  SCIP_RETCODE retcode;

  /* Iterate through each problem in the problems directory and make sure a
     valid solution structure exists for each problem. */
  std::cout << "[INFO]: "
            << "ImitationMILP: "
            << "Validating solutions directory structure."
            << "\n";
  for (auto& problem : fs::directory_iterator(fs::path(problems_path))) {
    /* not a valid problem file, skip. */
    if (fs::extension(problem.path()) != ".lp") {
      continue;
    }

    fs::path problem_name = problem.path().stem();
    fs::path solutions_dir = fs::path(solutions_path) / problem_name;

    /* If the solution directory did not exist, we create it and generate a
     * solution. */
    if (fs::create_directory(solutions_dir)) {
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
      retcode = SolveSCIP(problem_name.string(), problems_path,
                          solutions_dir.string(), NULL, NULL);
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

  /* Run the train loop. */
  Feat feat;
  
  /* Create the model. */
  RankNetModel model(feat.GetNumFeatures(), model_path, prev_model);
  success = model.Init();
  if (!success) {
    return success;
  }

  fs::path train_file = fs::path(problems_path) / (problems_path + ".train");
  fs::path valid_file = fs::path(problems_path) / (problems_path + ".valid");

  /* If there was no previous model, train an initial model with the oracle
     scorer as behavioral cloning. */
  if (prev_model == "") {
    std::cerr << "[INFO]: "
              << "ImitationMILP: "
              << "Previous model was not specified, training a new model and "
                 "saving to file: " << model_path << "\n";

    for (auto& problem : fs::directory_iterator(fs::path(problems_path))) {
      /* not a valid problem file, skip. */
      if (fs::extension(problem.path()) != ".lp") {
        continue;
      }

      fs::path problem_name = problem.path().stem();
      fs::path solutions_dir = fs::path(solutions_path) / problem_name;

      retcode = CreateNewSCIP();
      if (retcode != SCIP_OKAY) {
        success = false;
        break;
      }

      /* Create the oracle. */
      Oracle oracle(scip_, solutions_dir.string());

      /* Create the data collector. */
      RankedPairsCollector dc(scip_, train_file.string(), valid_file.string(),
                              &oracle, &feat);
      EventhdlrCollectData event_handler(scip_, &dc);

      /* Create the oracle node selector. */
      OracleScorer scorer(scip_, &oracle);
      NodeselPolicy node_selector(scip_, &scorer);

      /* Solve current SCIP instance. */
      retcode = SolveSCIP(problem_name.string(), problems_path, std::string(""),
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

    /* Train the initial model. */
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
  for (int i = 0; i < num_iters; ++i) {
    std::cerr << "[INFO]: " << "Starting train iteration " << i << "\n";

    for (auto& problem : fs::directory_iterator(fs::path(problems_path))) {
      /* not a valid problem file, skip. */
      if (fs::extension(problem.path()) != ".lp") {
        continue;
      }

      fs::path problem_name = problem.path().stem();
      fs::path solutions_dir = fs::path(solutions_path) / problem_name;

      retcode = CreateNewSCIP();
      if (retcode != SCIP_OKAY) {
        success = false;
        break;
      }

      /* Create the oracle. */
      Oracle oracle(scip_, solutions_dir.string());

      /* Create the data collector. */
      RankedPairsCollector dc(scip_, train_file.string(), valid_file.string(),
                              &oracle, &feat);
      EventhdlrCollectData event_handler(scip_, &dc);

      /* Create the node selector. */
      PythonScorer scorer(scip_, &model, &feat);
      NodeselPolicy node_selector(scip_, &scorer);

      /* Solve current SCIP instance. */
      retcode = SolveSCIP(problem_name.string(), problems_path, std::string(""),
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

    /* Train the initial model. */
    success = model.Train(train_file.string(), valid_file.string(), num_epochs,
                batch_size);
    if (!success) {
      return success;
    }
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