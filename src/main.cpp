/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * @file main.cpp
 * @brief ImitationMILP main file
 * @author Stephanie Ding
 *
 */

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include "imitation_milp.hpp"

/**
 * Main function
 * @param argc Argument count
 * @param argv Argument vector
 * @return 0
 */
int main(int argc, char** argv) {
  try {
    /** Define and parse the program options */
    bool is_train;
    std::string settings_file;
    std::string problems_path;
    std::string output_path;
    std::string train_path;
    std::string valid_path;
    // std::string solutions_path;
    std::string model_path;
    std::string prev_model;
    int num_iters;
    int num_epochs;

    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help,h", "produce help message")
      ("settings,x", po::value<std::string>(&settings_file)->default_value("scipparams.set", "SCIP parameters/settings file"))
      ("problems_path,p", po::value<std::string>(&problems_path)->default_value(""), "problem file or directory of files to be solved")
      ("output_path,o", po::value<std::string>(&output_path)->default_value(""), "path to save solutions to")
      ("train,t", po::bool_switch(&is_train)->default_value(false), "run in training mode")
      ("train_path,f", po::value<std::string>(&train_path)->default_value(""), "directory of training problems")
      ("valid_path,v", po::value<std::string>(&valid_path)->default_value(""), "directory of validation probelms")
      // ("solutions_path,s", po::value<std::string>(&solutions_path)->default_value(""), "directory containing solutions to the input problems (for training mode)")
      ("model_path,m", po::value<std::string>(&model_path)->default_value(""), "path of trained model (solve mode) or path to save trained models to (for training mode)")
      ("prev_model,w", po::value<std::string>(&prev_model)->default_value(""), "previous model to continue training on (for training mode)")
      ("num_iters,i", po::value<int>(&num_iters)->default_value(5), "number of DAgger iterations (for training mode)")
      ("num_epochs,e", po::value<int>(&num_epochs)->default_value(5), "number of model training epochs (for training mode)");

    po::variables_map vm;
    try {
      /* This can throw an exception on failure. */
      po::store(po::parse_command_line(argc, argv, desc), vm);

      /* Print help options. */
      if (vm.count("help")) {
        std::cout << "ImitationMILP" << std::endl << desc << std::endl;
        return EXIT_SUCCESS;
      }

      /* Will throw on error, do so after help in case there are any problems.
       */
      po::notify(vm);
    } catch (po::error& e) {
      std::cerr << "[ERROR]: " << e.what() << std::endl << std::endl;
      std::cerr << desc << std::endl;
      return EXIT_FAILURE;
    }

    /* Create the IMILP object */
    std::cout << "Starting ImitationMILP..."
              << "\n\n";
    imilp::ImitationMILP im(settings_file);

    /* TRAINING MODE */
    if (is_train) {
      /* Check required arguments. */
      if (model_path == "") {
        std::cerr
            << "[ERROR]: Model save path must be specified in training mode."
            << "\n\n";
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
      }

      if (train_path == "") {
        std::cerr << "[ERROR]: Path to training problems must be specified."
                  << "\n\n";
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
      }

      if (valid_path == "") {
        std::cerr << "[ERROR]: Path to validation problems must be specified."
                  << "\n\n";
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
      }

      /* Train */
      if (!im.Train(train_path, valid_path, model_path, prev_model,
                    num_iters, num_epochs, 32 /* batch size */)) {
        std::cerr << "[ERROR]: ImitationMILP encountered an error."
                  << "\n";
        return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;

    } else {
      /* SOLVE MODE */
      
      /* Check all required command-line arguments were passed. */
      if (problems_path == "") {
        std::cerr << "[ERROR]: Path to input problems must be specified."
                  << "\n\n";
        std::cerr << desc << std::endl;
        return EXIT_FAILURE;
      }

      /* Check required arguments. */
      // if (output_path == "") {
      //   std::cerr << "[ERROR]: Path to save solutions to must be specified."
      //             << "\n\n";
      //   std::cerr << desc << std::endl;
      //   return EXIT_FAILURE;
      // }

      /* Solve */
      if (!im.Solve(problems_path, output_path, model_path)) {
        std::cerr << "[ERROR]: ImitationMILP encountered an error."
                  << "\n";
        return EXIT_FAILURE;
      }

      return EXIT_SUCCESS;
    }

  } catch (std::exception& e) {
    std::cerr << "[ERROR]: Unhandled exception reached the top of main: "
              << e.what() << ", application will now exit." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}