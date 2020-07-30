#!/usr/bin/env python3

import sys
from pathlib import Path
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

class RankNetModule(nn.Module):
    def __init__(self, inputs, hidden_size, outputs, layers=1, dropout=None, batchnorm=False):
        super(RankNetModule, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(inputs, hidden_size),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Dropout(p=0.2),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_size, hidden_size),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_size, hidden_size),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_size, hidden_size),
        #     # # nn.BatchNorm1d(hidden_size),
        #     # nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.2),
        #     nn.Linear(hidden_size, outputs),
        # )
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.2),
            nn.Linear(hidden_size, outputs),
        )

        self.model.double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1.double())
        result_2 = self.model(input_2.double())
        pred = self.sigmoid(result_1 - result_2)
        return pred


class RankNet:
    '''
    Class that encapsulates a Keras model. An instance of this will be created
    on the C++ side.
    '''

    def __init__(self, model_file, input_dim, prev_model=None, is_gpu=False):
        '''
        Constructor. Initialize with either CPU or GPU session as specified.
        '''
        assert ('linux' in sys.platform), "This code runs on Linux only."

        self.model_file = model_file
        self.prev_model = prev_model
        self.input_dim = input_dim
        self.hidden_size = 128
        self.outputs = 1
        self.model = None
        self.iters = 0

        self.device = None
        if is_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if prev_model and (not prev_model == ""):
            self.load_model()

    '''
    Load an existing model
    '''
    def load_model(self):
        assert(self.prev_model and Path(self.prev_model).is_file()
               ), "Model {} could not be loaded.".format(self.prev_model)
         
        self.model = RankNetModule(self.input_dim, self.hidden_size, self.outputs).to(self.device)
        self.model.load_state_dict(torch.load(self.prev_model))
        # self.model.eval()


    def save_model(self, model_file):
        device = torch.device('cpu')
        # model = RankNetModuleEval(self.input_dim, self.hidden_size, self.outputs).to(device)
        # model.load_state_dict(self.model.state_dict())

        self.model.to(device)
        self.model.eval()
        e1 = torch.rand(1, self.input_dim).to(device)
        e2 = torch.rand(1, self.input_dim).to(device)
        traced_script_module = torch.jit.trace(self.model, (e1, e2))

        traced_script_module.save(model_file)

        output = traced_script_module(e1, e2)

        self.model.to(self.device)

    def load_data(self, dirs, filename=None):
        n_rows = 0

        for d, should_sample in dirs:
            files = Path(d).glob("*.data")
            for file in files:
                if should_sample:
                    n_rows += int(0.1 * (file_len(file) - 1))
                else:
                    n_rows += int(file_len(file) - 1)

        data = np.zeros((n_rows, 2 * self.input_dim + 2))
        i = 0

        for d, should_sample in dirs:
            files = Path(d).glob("*.data")
            for file in files:
                curr = np.genfromtxt(file, delimiter=",", skip_header=1)

                if curr.ndim == 1:
                    continue

                if should_sample:
                    n = int(0.1 * np.size(curr, 0))
                    sample = curr[curr[:,0].argsort()][:n, :]

                    data[i:i+n, :] = sample 
                    i += n
                else:
                    n = np.size(curr, 0)
                    data[i:i+n, :] = curr
                    i += n

        if filename is not None:
            np.save(filename, data)

        return data

    def train(self, train_dirs, valid_dirs, test_dirs, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        # Load data
        # train_data = np.empty(2 * self.input_dim + 2)
        # for train_dir in train_dirs:
        #     train_files = Path(train_dir).glob("*.data")
        #     for file in train_files:
        #         data = np.genfromtxt(file, delimiter=",", skip_header=1)
        #         if data.ndim == 1:
        #             continue
        #         train_data = np.row_stack((train_data, data))

        # valid_data = np.empty(2 * self.input_dim + 2)
        # for valid_dir in valid_dirs:
        #     valid_files = Path(valid_dir).glob("*.data")
        #     for file in valid_files:
        #         data = np.genfromtxt(file, delimiter=",", skip_header=1)
        #         if data.ndim == 1:
        #             continue
        #         valid_data = np.row_stack((valid_data, data))

        train_data = self.load_data(train_dirs)
        valid_data = self.load_data(valid_dirs)

        # train_data = np.load('train.npy')
        # valid_data = np.load('valid.npy')
        # test_data = np.load('test.npy')

        # Separate training data into components
        train_weights = torch.from_numpy(train_data[:, 0])
        train_y = torch.from_numpy(train_data[:, -1].astype(int))
        train_X1 = torch.from_numpy(train_data[:, 1:self.input_dim + 1])
        train_X2 = torch.from_numpy(train_data[:, self.input_dim + 1:-1])

        # Fit scaler using training data
    
        # self.scaler = StandardScaler()
        # train_X = np.vstack([train_X1, train_X2])
        # self.scaler.fit(train_X)

        # joblib.dump(self.scaler, model_file + ".scaler")

        # Separate validation data into components
        valid_weights = torch.from_numpy(valid_data[:, 0]).double().to(self.device)
        valid_y = torch.from_numpy(valid_data[:, -1].astype(int)).double().to(self.device)
        valid_X1 = torch.from_numpy(valid_data[:, 1:self.input_dim + 1]).double().to(self.device)
        valid_X2 = torch.from_numpy(valid_data[:, self.input_dim + 1:-1]).double().to(self.device)
        valid_criterion = nn.BCELoss(weight=valid_weights)

        # Scale training and validation data
        # train_X1 = self.scaler.transform(train_X1)
        # train_X2 = self.scaler.transform(train_X2)
        # valid_X1 = self.scaler.transform(valid_X1)
        # valid_X2 = self.scaler.transform(valid_X2)

        N_train = train_data.shape[0]
        N_valid = valid_data.shape[0]

        print("Train on {}, test on {}".format(N_train, N_valid))

        # Train model
        learning_rate = 0.01
        self.model = RankNetModule(self.input_dim, self.hidden_size, self.outputs).to(self.device)
        optimizer = optim.Adadelta(self.model.parameters(), lr = learning_rate)
       
        self.model.train()
        
        train_loss = torch.zeros(1)
        valid_loss = torch.zeros(1)
        prev_valid_loss = float("inf")

        for epoch in range(num_epochs):
            idx_train = torch.randperm(N_train)

            train_X1 = train_X1[idx_train]
            train_X2 = train_X2[idx_train]
            train_y = train_y[idx_train]
            train_w = train_weights[idx_train]            

            cur_batch = 0

            train_acc = 0
            train_loss = 0.0
            n = 0

            for i in range(N_train // batch_size):
                batch_X1 = train_X1[cur_batch: cur_batch + batch_size].double().to(self.device)
                batch_X2 = train_X2[cur_batch: cur_batch + batch_size].double().to(self.device)
                batch_y = train_y[cur_batch: cur_batch + batch_size].double().to(self.device)
                batch_w = train_w[cur_batch: cur_batch + batch_size].double().to(self.device)
                train_criterion = nn.BCELoss(weight=batch_w)
                cur_batch += batch_size

                optimizer.zero_grad()
                batch_loss = torch.zeros(1)

                if len(batch_X1) > 0:
                    batch_pred = self.model(batch_X1, batch_X2)
                    batch_loss = train_criterion(batch_pred, batch_y)
                    train_loss += batch_loss.item()

                    batch_pred_binary = torch.tensor([1 if x >= 0.5 else 0 for x in batch_pred])
                    train_acc += (batch_pred_binary == batch_y).sum().data.numpy()/batch_pred.size()[0]

                batch_loss.backward(retain_graph=True)
                optimizer.step()
                n += 1

            train_loss /= n
            train_acc /= n

            # print(valid_X1.size(), valid_X2.size())

            with torch.no_grad():
                valid_pred = self.model(valid_X1, valid_X2)
                valid_loss = valid_criterion(valid_pred, valid_y)

                valid_pred_binary = torch.tensor([1 if x >= 0.5 else 0 for x in valid_pred])
                valid_acc = (valid_pred_binary == valid_y).sum().data.numpy()/valid_pred.size()[0]

                print('epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, valid loss: {:.4f}, valid acc: {:.4f}'
                      .format(epoch + 1, num_epochs, train_loss, train_acc, valid_loss.item(), valid_acc))

                # if valid_loss.item() - prev_valid_loss > 0.1:
                #     break

                prev_valid_loss = valid_loss.item()

                # save model
                model_file = self.model_file.split(".")[0] + "-" + str(epoch) + ".pt"
                self.prev_model = model_file
                self.save_model(model_file)

        # model_file = self.model_file.split(".")[0] + "-" + str(self.iters) + ".pt"
        # torch.save(self.model.state_dict(), model_file)

        # save model
        self.prev_model = self.model_file
        self.save_model(self.model_file)

        # self.prev_model = model_file
        # self.load_model()


    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
        # predict on the CPU
        device = torch.device('cpu')

        if self.prev_model is None or self.prev_model == "":
            self.prev_model = self.model_file

        if self.model is None:
            self.load_model()

        X1 = np.array(X1).reshape((1, self.input_dim))
        X1 = torch.from_numpy(X1).double().to(device)
        X2 = np.array(X2).reshape((1, self.input_dim))
        X2 = torch.from_numpy(X2).double().to(device)
    
        self.model.eval()
        pred = self.model(X1, X2).item()

        if (pred >= 0.5):
            return 1
        else:
            return 0


# train_dirs = [("data/hybrid_bids/bids_500/train/oracle", False),
#               ("data/hybrid_bids/bids_500/train/iter1", True),
#               ("data/hybrid_bids/bids_500/train/iter2", True),
#               ("data/hybrid_bids/bids_500/train/iter3", True)]
# valid_dirs = [("data/hybrid_bids/bids_500/valid/oracle", False),
#               ("data/hybrid_bids/bids_500/valid/iter1", True),
#               ("data/hybrid_bids/bids_500/valid/iter2", True),
#               ("data/hybrid_bids/bids_500/valid/iter3", True)]


train_dirs = [("data/mvc/data/train/oracle", False),
              ("data/mvc/data/train/iter1", True),
              ("data/mvc/data/train/iter2", True),
              ("data/mvc/data/train/iter3", True)]
              # ("data/hybrid_bids/bids_850/train/iter1", True)]
              # ("data/hybrid_bids/bids_850/train/iter2", True),
              # ("data/hybrid_bids/bids_850/train/iter3", True)]
#               ("data/hybrid_bids/bids_500/train/iter2", True),
#               ("data/hybrid_bids/bids_500/train/iter3", True)]
valid_dirs = [("data/mvc/data/valid/oracle", False),
              ("data/mvc/data/valid/iter1", True),
              ("data/mvc/data/valid/iter2", True),
              ("data/mvc/data/valid/iter3", True)]
              # ("data/hybrid_bids/bids_850/valid/iter1", True)]
              # ("data/hybrid_bids/bids_850/valid/iter2", True),
              # ("data/hybrid_bids/bids_850/valid/iter3", True)]
              # ("data/hybrid_bids/bids_500/valid/iter2", True),
              # ("data/hybrid_bids/bids_500/valid/iter3", True)]

test_dirs = [("data/mvc/data/test/oracle", False)]
              # ("data/hybrid_bids/bids_500/test/iter1", True),
              # ("data/hybrid_bids/bids_500/test/iter2", True)]


m = RankNet("models/mvc/iter3/mvc_iter3.pt", 26, "")
m.train(train_dirs, valid_dirs, test_dirs, 100, 32)
# m.test(test_dirs)