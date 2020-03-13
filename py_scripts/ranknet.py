import sys
from pathlib import Path
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RankNetModule(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(RankNetModule, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Linear(hidden_size, outputs),
            #nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        result_1 = self.model(input_1) #预测input_1得分
        result_2 = self.model(input_2) #预测input_2得分
        pred = self.sigmoid(result_1 - result_2) #input_1比input_2更相关概率
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
        self.iters = 7

        if prev_model and (not prev_model == ""):
            self.load_model()

    def load_model(self):
        assert(self.prev_model and Path(self.prev_model).is_file()
               ), "Model {} could not be loaded.".format(self.prev_model)
         
        self.model = RankNetModule(self.input_dim, self.hidden_size, self.outputs).to(device)
        self.model.load_state_dict(torch.load(self.prev_model))


    def train(self, train_dir, valid_dir, num_epochs, batch_size):
        pass


    def train_multi(self, train_dirs, valid_dirs, num_epochs, batch_size):
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
        valid_weights = torch.from_numpy(valid_data[:, 0]).float().to(device)
        valid_y = torch.from_numpy(valid_data[:, -1].astype(int)).float().to(device)
        valid_X1 = torch.from_numpy(valid_data[:, 1:self.input_dim + 1]).float().to(device)
        valid_X2 = torch.from_numpy(valid_data[:, self.input_dim + 1:-1]).float().to(device)
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
        self.model = RankNetModule(self.input_dim, self.hidden_size, self.outputs).to(device)
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
                batch_X1 = train_X1[cur_batch: cur_batch + batch_size].float().to(device)
                batch_X2 = train_X2[cur_batch: cur_batch + batch_size].float().to(device)
                batch_y = train_y[cur_batch: cur_batch + batch_size].float().to(device)
                batch_w = train_w[cur_batch: cur_batch + batch_size].float().to(device)
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

                model_file = self.model_file.rstrip(
                    ".h5") + "-" + str(self.iters) + ".h5"
                torch.save(self.model.state_dict(), model_file)

        self.prev_model = model_file
        self.load_model()


    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
        # predict on the CPU
        if self.prev_model is None or self.prev_model == "":
            self.prev_model = self.model_file

        if self.model is None:
            self.load_model()

        X1 = np.array(X1).reshape((1, self.input_dim))
        X1 = torch.from_numpy(X1).float().to(device)
        X2 = np.array(X2).reshape((1, self.input_dim))
        X2 = torch.from_numpy(X2).float().to(device)
    
        self.model.eval()
        pred = self.model(X1, X2).item()

        if (pred >= 0.5):
            return 1
        else:
            return 0


# train_dirs = ["data/hybrid_bids/bids_500/train_small/oracle",
#               "data/hybrid_bids/bids_500/train_small/iter1",
#               "data/hybrid_bids/bids_500/train_small/iter2",
#               "data/hybrid_bids/bids_500/train_small/iter3",
#               "data/hybrid_bids/bids_500/train_small/iter4",
#               "data/hybrid_bids/bids_500/train_small/iter5",
#               "data/hybrid_bids/bids_500/train_small/iter6",
#               "data/hybrid_bids/bids_500/train_small/iter7"]
# valid_dirs = ["data/hybrid_bids/bids_500/valid_small/oracle",
#               "data/hybrid_bids/bids_500/valid_small/iter1",
#               "data/hybrid_bids/bids_500/valid_small/iter2",
#               "data/hybrid_bids/bids_500/valid_small/iter3",
#               "data/hybrid_bids/bids_500/valid_small/iter4",
#               "data/hybrid_bids/bids_500/valid_small/iter5",
#               "data/hybrid_bids/bids_500/valid_small/iter6",
#               "data/hybrid_bids/bids_500/valid_small/iter7"]
# # "data/hybrid_bids/bids_530/valid/scaling"]

# m = RankNet("models/hybrid_bids_500_small_pt.h5", 26, "")
# m.train_multi(train_dirs, valid_dirs, 100, 32)
# print(m.predict([1] * 26, [0] * 26))


# train_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_train/data/"
# valid_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_valid/data/"
# m = RankNet("models/3dp_2.h5", 26, "")
# m.train(train_path, valid_path, 100, 32)
# m.predict([0] * 26, [1] * 26)

# print("TRAIN:")
# for file in Path(train_path).glob("*.data"):
#     data = np.genfromtxt(file, delimiter=",", skip_header=1)
#     weights = data[:, 0]  # target
#     y = data[:, -1]
#     X1 = data[:, 1:27]
#     X2 = data[:, 27:-1]

#     y = y.reshape((len(y), 1))

#     s1 = m.model.predict([X1, X2])
#     # print(s1)
#     # print(y)
#     # s2 = m.model.predict([X2, X1])
#     # s2 = s2.reshape((len(s2), 1))
#     # s3 = (s1 > s2).astype(int)
#     # print(y.shape)
#     # print(s1.shape)
#     s1 = s1.reshape((len(s1), 1))
#     s1 = (s1 > 0.5).astype(int)
#     # print(s1)
#     errs = np.sum(s1 != np.array(y))
#     print("{}, samples={}, wrong={}".format(file.stem, len(y), errs))

# print("\nVALID:")
# for file in Path(valid_path).glob("*.data"):
#     data = np.genfromtxt(file, delimiter=",", skip_header=1)
#     weights = data[:, 0]  # target
#     y = data[:, -1]
#     X1 = data[:, 1:27]
#     X2 = data[:, 27:-1]

#     s1 = m.model.predict([X1, X2])
#     # s1 = s1.reshape((len(s1), 1))
#     # print(s1)
#     # print(y)
#     # s2 = m.model.predict([X2, X1])
#     # s2 = s2.reshape((len(s2), 1))
#     y = y.reshape((len(y), 1))
#     # s3 = (s1 > s2).astype(int)
#     s1 = s1.reshape((len(s1), 1))
#     s1 = (s1 > 0.5).astype(int)
#     # print(s1)
#     errs = np.sum(s1 != np.array(y))
#     print("{}, samples={}, wrong={}".format(file.stem, len(y), errs))
