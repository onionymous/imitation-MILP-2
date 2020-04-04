/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*    This file is part of the program and library ImitationMILP             */
/*    Copyright (C) 2018 Caltech                                             */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   ranknet_model.hpp
 * @brief  C++ interface to the Python RankNet model.
 * @author Stephanie Ding
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#ifndef RANKNET_MODEL_HPP
#define RANKNET_MODEL_HPP

#include <string>
#include <utility>
#include <vector>

#include <boost/python.hpp>
#include <python3.6m/Python.h>
#include <torch/script.h>

namespace imilp {

/** Python scorer. */
class RankNetModel {
 public:
  RankNetModel(int input_dim, const std::string& model_path,
               const std::string& prev_model)
      : input_dim_(input_dim),
        model_path_(model_path),
        prev_model_(prev_model) {}
  ~RankNetModel() {};

  /** Python initialization. */
  bool Init(bool is_gpu);

  /** Train model. */
  bool Train(const std::string& train_path, const std::string& valid_path,
             int num_epochs, int batch_size);

  /** Use model to get a prediction. */
  int Predict(const std::vector<double>& x1, const std::vector<double>& x2);

 private:
  int input_dim_;
  std::string model_path_;
  std::string prev_model_;

  boost::python::object py_file_;
  boost::python::object RankNet_py_;
  torch::jit::script::Module module_;

};

}  // namespace imilp

#endif
