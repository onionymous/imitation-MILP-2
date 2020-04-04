#!/usr/bin/env python3

import sys
from pathlib import Path
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class RankNetModule(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(RankNetModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, outputs),
        )
        self.model.double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1.double())
        result_2 = self.model(input_2.double())
        pred = self.sigmoid(result_1 - result_2)
        return pred


'''
Modified version of the model for evaluation
'''
class RankNetModuleEval(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(RankNetModuleEval, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, outputs),
        )
        self.model.double()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1.view(1, -1).double())
        result_2 = self.model(input_2.view(1, -1).double())
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
        self.hidden_size = 10
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


    def save_model(self, model_file):
        device = torch.device('cpu')
        model = RankNetModuleEval(self.input_dim, self.hidden_size, self.outputs).to(device)
        model.load_state_dict(self.model.state_dict())

        model.to(device)
        e1 = torch.rand(1, self.input_dim).to(device)
        e2 = torch.rand(1, self.input_dim).to(device)
        traced_script_module = torch.jit.trace(model, (e1, e2))

        traced_script_module.save(model_file)

    def train(self, train_dirs, valid_dirs, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        # Load data
        train_data = np.empty(2 * self.input_dim + 2)
        for train_dir in train_dirs:
            train_files = Path(train_dir).glob("*.data")
            for file in train_files:
                data = np.genfromtxt(file, delimiter=",", skip_header=1)
                if data.ndim == 1:
                    continue
                train_data = np.row_stack((train_data, data))

        valid_data = np.empty(2 * self.input_dim + 2)
        for valid_dir in valid_dirs:
            valid_files = Path(valid_dir).glob("*.data")
            for file in valid_files:
                data = np.genfromtxt(file, delimiter=",", skip_header=1)
                if data.ndim == 1:
                    continue
                valid_data = np.row_stack((valid_data, data))

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
        learning_rate = 0.2
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

            train_loss = 0.0

            cur_batch = 0
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

                batch_loss.backward(retain_graph=True)
                optimizer.step()

            with torch.no_grad():
                valid_pred = self.model(valid_X1, valid_X2)
                valid_loss = valid_criterion(valid_pred, valid_y)

                print('epoch [{}/{}], train loss: {:.4f}, valid loss: {:.4f}'
                      .format(epoch + 1, num_epochs, train_loss, valid_loss.item()))

                if valid_loss.item() - prev_valid_loss > 0.1:
                    break

                prev_valid_loss = valid_loss.item()

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


train_dirs = ["data/hybrid_bids/bids_500/train/oracle",
              "data/hybrid_bids/bids_500/train/iter1"]
valid_dirs = ["data/hybrid_bids/bids_500/valid/oracle",
              "data/hybrid_bids/bids_500/valid/iter1"]

m = RankNet("models/bids_500-1.pt", 26, "")
m.train(train_dirs, valid_dirs, 100, 32)
print(m.predict([1] * 26, [0] * 26))


# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         print("Usage:")
#         print("\t{} [log file]".format(sys.argv[0]))
#     else:
#         parse_logs(sys.argv[1])
