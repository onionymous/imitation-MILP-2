/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ranked_pairs_collector.cpp
 * @brief  Data collector class that collects data for training a ranking model
           with retrospective knowledge of node optimality, and writes the
           collected data to a .csv file.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "ranknet_model.hpp"

#include <cassert>
#include <iostream>

#include <boost/filesystem.hpp>

namespace imilp {

namespace bp = boost::python;
namespace fs = boost::filesystem;

bool RankNetModel::Init() {
  bool success = true;
  try {
    Py_Initialize();
    fs::path working_dir = fs::absolute("./").normalize();
    working_dir /= "py_scripts";
    // std::cout << working_dir << "\n";
    PyObject* sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0,
                  PyUnicode_FromString(working_dir.string().c_str()));

    /* Import Python module and create an instance of the class. */
    py_file_ = bp::import("ranknet");
    RankNet_py_ = py_file_.attr("RankNet")(
        model_path_, input_dim_,
        prev_model_);

  } catch (bp::error_already_set&) {
    success = false;

    /* Handle error within embedded Python. */
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    bp::handle<> hType(ptype);
    bp::object extype(hType);
    bp::handle<> hTraceback(ptraceback);
    bp::object traceback(hTraceback);

    /* Extract error message */
    std::string error_msg = bp::extract<std::string>(pvalue);

    /* Extract line number (top entry of call stack).
       Other levels of call stack can be extracted by processing
       traceback.attr("tb_next") recurently. */
    long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
    std::string filename = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
    std::string funcname = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_name"));

    /* Print out the error details. */
    std::cout << "[ERROR]: Python: Error on line: " << lineno
              << ", function: " << funcname << "\n";
    std::cout << error_msg << "\n\n";
  }

  return success;
}

/** Train model. */
bool RankNetModel::Train(const std::string& train_file,
                         const std::string& valid_file, int num_epochs,
                         int batch_size) {
  bool success = true;
  try {
    /* Call the functions of the Python class to train. */
    fs::path train_file_abs = fs::absolute(train_file).normalize();
    fs::path valid_file_abs = fs::absolute(valid_file).normalize();
    bp::object result = RankNet_py_.attr("train")(train_file_abs.string(),
                                                  valid_file_abs.string(),
                                                  num_epochs, batch_size);
    /* Handle error within embedded Python. */
  } catch (bp::error_already_set&) {
    success = false;

    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    bp::handle<> hType(ptype);
    bp::object extype(hType);
    bp::handle<> hTraceback(ptraceback);
    bp::object traceback(hTraceback);

    /* Extract error message */
    std::string error_msg = bp::extract<std::string>(pvalue);

    /* Extract line number (top entry of call stack).
       Other levels of call stack can be extracted by processing
       traceback.attr("tb_next") recurently. */
    long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
    std::string filename = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
    std::string funcname = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_name"));

    /* Print out the error details. */
    std::cout << "[ERROR]: Python: Error on line: " << lineno
              << "function: " << funcname << "\n";
    std::cout << error_msg << "\n";
  }

  return success;
}

/** Train model. */
std::pair<double, double> RankNetModel::Predict(const std::vector<double>& x1,
                                                const std::vector<double>& x2) {
  try {
    /* Convert vectors to Python lists. */
    bp::list x1_list;
    bp::list x2_list;

    for (int i = 0; i < input_dim_; ++i) {
      x1_list.append(x1[i]);
      x2_list.append(x2[i]);
    }

    /* Call the functions of the Python class to get the prediction. */
    bp::object result =
        RankNet_py_.attr("predict")(x1_list, x2_list);

    double score1 = bp::extract<double>(result[0]);
    double score2 = bp::extract<double>(result[1]);

    /* Return the prediction. */
    return std::make_pair(score1, score2);

    /* Handle error within embedded Python. */
  } catch (bp::error_already_set&) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    bp::handle<> hType(ptype);
    bp::object extype(hType);
    bp::handle<> hTraceback(ptraceback);
    bp::object traceback(hTraceback);

    /* Extract error message */
    std::string error_msg = bp::extract<std::string>(pvalue);

    /* Extract line number (top entry of call stack).
       Other levels of call stack can be extracted by processing
       traceback.attr("tb_next") recurently. */
    long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
    std::string filename = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
    std::string funcname = bp::extract<std::string>(
        traceback.attr("tb_frame").attr("f_code").attr("co_name"));

    /* Print out the error details. */
    std::cout << "[ERROR]: Python: Error on line: " << lineno
              << "function: " << funcname << "\n";
    std::cout << error_msg << "\n";
  }

  /* If an error occurred. */
  return std::make_pair(-1, -1);
}

}  // namespace imilp