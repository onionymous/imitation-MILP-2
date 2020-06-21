/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ranknet_model.cpp
 * @brief  C++ interface to the Python RankNet model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include "ranknet_model.hpp"

#include <cassert>
#include <iostream>

#include <boost/filesystem.hpp>

#include <dlfcn.h>

namespace imilp {

namespace bp = boost::python;
namespace fs = boost::filesystem;

std::string handle_pyerror() {
    using namespace boost::python;
    using namespace boost;

    PyObject *exc,*val,*tb;
    object formatted_list, formatted;
    PyErr_Fetch(&exc,&val,&tb);
    handle<> hexc(exc),hval(allow_null(val)),htb(allow_null(tb)); 
    object traceback(import("traceback"));
    if (!tb) {
        object format_exception_only(traceback.attr("format_exception_only"));
        formatted_list = format_exception_only(hexc,hval);
    } else {
        object format_exception(traceback.attr("format_exception"));
        formatted_list = format_exception(hexc,hval,htb);
    }
    formatted = str("\n").join(formatted_list);
    return extract<std::string>(formatted);
}

bool RankNetModel::Init(bool is_gpu) {
  bool success = true;

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module_ = torch::jit::load(prev_model_);
    // module_.to(at::kCUDA);
  }
  catch (const c10::Error& e) {
    std::cerr << "[ERROR]: Error loading model error, terminating." << "\n";
    success = false;
  }

  // try {
  //   void *const libpython_handle = dlopen("libpython3.7m.so", RTLD_LAZY | RTLD_GLOBAL);
  //   Py_Initialize();
  //   fs::path working_dir = fs::absolute("./").normalize();
  //   working_dir /= "py_scripts";
  //   // std::cout << working_dir << "\n";
  //   PyObject* sys_path = PySys_GetObject("path");
  //   PyList_Insert(sys_path, 0,
  //                 PyUnicode_FromString(working_dir.string().c_str()));

  //   /* Import Python module and create an instance of the class. */
  //   py_file_ = bp::import("ranknet");
  //   RankNet_py_ = py_file_.attr("RankNet")(
  //       model_path_, input_dim_,
  //       prev_model_, is_gpu);

  // } catch (bp::error_already_set&) {
  //   success = false;
  //   if (PyErr_Occurred()) {
  //      std::string msg = handle_pyerror(); 
  //      std::cout << "[ERROR]: Python error: " << msg << "\n";
  //   }
  //   bp::handle_exception();
  //   PyErr_Clear();

  //   // /* Handle error within embedded Python. */
  //   // PyObject *ptype, *pvalue, *ptraceback;
  //   // PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  //   // bp::handle<> hType(ptype);
  //   // bp::object extype(hType);
  //   // bp::handle<> hTraceback(ptraceback);
  //   // bp::object traceback(hTraceback);
  //   // 
  //   // if (pvalue) {
  //   // 	/* Extract error message */
  //   // 	std::string error_msg = bp::extract<std::string>(pvalue);
  //   // /* Extract line number (top entry of call stack).
  //   //    Other levels of call stack can be extracted by processing
  //   //    traceback.attr("tb_next") recurently. */
  //   // long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
  //   // std::string filename = bp::extract<std::string>(
  //   //     traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
  //   // std::string funcname = bp::extract<std::string>(
  //   //     traceback.attr("tb_frame").attr("f_code").attr("co_name"));

  //   // /* Print out the error details. */
  //   // std::cout << "[ERROR]: Python: Error on line: " << lineno
  //   //           << ", function: " << funcname << "\n";
  //   // std::cout << error_msg << "\n\n";
  //   // }

  //   std::cout << "[ERROR]: Python error, terminating." << "\n";
  // }

  return success;
}

/** Train model. */
bool RankNetModel::Train(const std::string& train_path,
                         const std::string& valid_path, int num_epochs,
                         int batch_size) {
  bool success = false; // true;
  // try {
  //   /* Call the functions of the Python class to train. */
  //   fs::path train_path_abs = fs::absolute(train_path).normalize();
  //   fs::path valid_path_abs = fs::absolute(valid_path).normalize();
  //   bp::object result = RankNet_py_.attr("train")(train_path_abs.string(),
  //                                                 valid_path_abs.string(),
  //                                                 num_epochs, batch_size);
  //   /* Handle error within embedded Python. */
  // } catch (bp::error_already_set&) {
  //   success = false;

  //   // PyObject *ptype, *pvalue, *ptraceback;
  //   // PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  //   // bp::handle<> hType(ptype);
  //   // bp::object extype(hType);
  //   // bp::handle<> hTraceback(ptraceback);
  //   // bp::object traceback(hTraceback);

  //   // /* Extract error message */
  //   // std::string error_msg = bp::extract<std::string>(pvalue);

  //   // /* Extract line number (top entry of call stack).
  //   //    Other levels of call stack can be extracted by processing
  //   //    traceback.attr("tb_next") recurently. */
  //   // long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
  //   // std::string filename = bp::extract<std::string>(
  //   //     traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
  //   // std::string funcname = bp::extract<std::string>(
  //   //     traceback.attr("tb_frame").attr("f_code").attr("co_name"));

  //   // /* Print out the error details. */
  //   // std::cout << "[ERROR]: Python: Error on line: " << lineno
  //   //           << "function: " << funcname << "\n";
  //   // std::cout << error_msg << "\n";
  // }

  return success;
}

/** Train model. */
int RankNetModel::Predict(const std::vector<double>& x1,
                          const std::vector<double>& x2) {
  assert(x1.size() == input_dim_);
  assert(x2.size() == input_dim_);

  // try {
  //   /* Convert vectors to Python lists. */
  //   bp::list x1_list;
  //   bp::list x2_list;

  //   for (auto& x : x1) {
  //     x1_list.append(x);
  //   }

  //   for (auto& x : x2) {
  //     x2_list.append(x);
  //   }

  //   // for (int i = 0; i < input_dim_; ++i) {
  //   //   x1_list.append(x1[i]);
  //   //   x2_list.append(x2[i]);
  //   // }

  //   /* Call the functions of the Python class to get the prediction. */
  //   bp::object result_obj =
  //       RankNet_py_.attr("predict")(x1_list, x2_list);

  //   // double score1 = bp::extract<double>(result[0]);
  //   // double score2 = bp::extract<double>(result[1]);

  //   int result = bp::extract<int>(result_obj);

  //   /* Return the prediction. */
  //   // return std::make_pair(score1, score2);
  //   return result;

  //   /* Handle error within embedded Python. */
  // } catch (bp::error_already_set&) {
  //   PyObject *ptype, *pvalue, *ptraceback;
  //   PyErr_Fetch(&ptype, &pvalue, &ptraceback);

  //   bp::handle<> hType(ptype);
  //   bp::object extype(hType);
  //   bp::handle<> hTraceback(ptraceback);
  //   bp::object traceback(hTraceback);

  //   /* Extract error message */
  //   std::string error_msg = bp::extract<std::string>(pvalue);

  //   /* Extract line number (top entry of call stack).
  //      Other levels of call stack can be extracted by processing
  //      traceback.attr("tb_next") recurently. */
  //   long lineno = bp::extract<long>(traceback.attr("tb_lineno"));
  //   std::string filename = bp::extract<std::string>(
  //       traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
  //   std::string funcname = bp::extract<std::string>(
  //       traceback.attr("tb_frame").attr("f_code").attr("co_name"));

  //   /* Print out the error details. */
  //   std::cout << "[ERROR]: Python: Error on line: " << lineno
  //             << "function: " << funcname << "\n";
  //   std::cout << error_msg << "\n";
  // }

  //try {
    std::vector<torch::jit::IValue> inputs;
    // std::vector<torch::jit::IValue> tuple;

    inputs.push_back(torch::tensor(x1).reshape({1, -1}));
    inputs.push_back(torch::tensor(x2).reshape({1, -1}));
    // inputs.push_back(torch::ivalue::Tuple::create(tuple));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module_.forward(inputs).toTensor();
    // double pred = output[0].item<double>();
    double pred = output[0].item<double>();
    // std::cout << pred << "\n";

    return (pred >= 0.5) ? 1 : 0;
  //} catch (const c10::Error& e) {
  /* If an error occurred. */
  // return std::make_pair(-1, -1);
  // return -1;
  //}
}

}  // namespace imilp
