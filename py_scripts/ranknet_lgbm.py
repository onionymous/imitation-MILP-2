import sys
import numpy as np
from pathlib import Path
import lightgbm as lgb
import joblib

NUM_CORES = 4


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
        self.model = None
        self.scaler = None
        self.iters = 0

        if prev_model and (prev_model != ""):
            self.load_model()

    def load_model(self):
        assert(self.prev_model and Path(self.prev_model).is_file()
               ), "Model {} could not be loaded.".format(self.prev_model)
        self.model = joblib.load(self.prev_model)
        # self.scaler = joblib.load(self.prev_model + ".scaler")

    def train(self, train_dir, valid_dir, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''
        # Load data
        train_q = []
        train_data = np.empty(2 * self.input_dim + 2)
        # for train_dir in train_dirs:
        train_files = Path(train_dir).glob("*.data")
        for file in train_files:
            data = np.genfromtxt(file, delimiter=",", skip_header=1)
            if data.ndim == 1:
                continue
            train_data = np.row_stack((train_data, data))
            train_q.append(data.shape[0])
        train_q = np.array(train_q)

        valid_q = []
        valid_data = np.empty(2 * self.input_dim + 2)
        # for valid_dir in valid_dirs:
        valid_files = Path(valid_dir).glob("*.data")
        for file in valid_files:
            data = np.genfromtxt(file, delimiter=",", skip_header=1)
            if data.ndim == 1:
                continue
            valid_data = np.row_stack((valid_data, data))
            valid_q.append(data.shape[0])
        valid_q = np.array(valid_q)

        # Separate training data into components
        train_weights = train_data[:, 0]  # target
        train_y = train_data[:, -1].astype(int)
        # train_X1 = train_data[:, 1:self.input_dim + 1]
        # train_X2 = train_data[:, self.input_dim + 1:-1]
        train_X = train_data[:, 1:-1]

        # Fit scaler using training data
        model_file = self.model_file.rstrip(
            ".h5") + "-" + str(self.iters) + ".h5"
        # # train_X = np.vstack([train_X1, train_X2])
        # self.scaler = StandardScaler()
        # self.scaler.fit(train_X)
        # joblib.dump(self.scaler, model_file + ".scaler")

        # Separate validation data into components
        valid_weights = valid_data[:, 0]  # target
        valid_y = valid_data[:, -1].astype(int)
        # valid_X1 = valid_data[:, 1:self.input_dim + 1]
        # valid_X2 = valid_data[:, self.input_dim + 1:-1]
        valid_X = valid_data[:, 1:-1]

        # Scale training and validation data
        # train_X1 = self.scaler.transform(train_X1)
        # train_X2 = self.scaler.transform(train_X2)
        # valid_X1 = self.scaler.transform(valid_X1)
        # valid_X2 = self.scaler.transform(valid_X2)
        # train_X = self.scaler.transform(train_X)
        # valid_X = self.scaler.transform(valid_X)

        print("Train on {}, test on {}".format(train_X.shape, valid_X.shape))

        # Train model
        self.model = lgb.LGBMRanker()
        self.model.fit(X=train_X, y=train_y, sample_weight=train_weights,
                       group=train_q, eval_set=[(valid_X, valid_y)],
                       eval_group=[valid_q], eval_sample_weight=valid_weights,
                       eval_at=[1, 3], early_stopping_rounds=20, verbose=True,
                       callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

        joblib.dump(self.model, model_file)

        self.prev_model = model_file
        self.load_model()

    def train_batch(self, train_dirs, valid_dirs, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        # Load data
        train_q = []
        train_weights = []
        train_X = []
        train_y = []
        for train_dir in train_dirs:
            train_files = Path(train_dir).glob("*.data")
            for file in train_files:
                f = open(file)
                f.readline()
                for line in f:
                    row = [float(x) for x in line.strip().split(",")]
                    train_X.append(row[1:self.input_dim + 1])
                    train_X.append(row[self.input_dim + 1:-1])
                    y = int(row[-1])
                    if y == 0:
                        train_y.append(0)
                        train_y.append(1)
                    else:
                        train_y.append(1)
                        train_y.append(0)
                    train_weights.append(row[0])
                    train_weights.append(row[0])
                    train_q.append(2)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        train_q = np.array(train_q)
        train_weights = np.array(train_weights)

        valid_q = []
        valid_weights = []
        valid_X = []
        valid_y = []
        for valid_dir in valid_dirs:
            valid_files = Path(valid_dir).glob("*.data")
            for file in valid_files:
                f = open(file)
                f.readline()
                for line in f:
                    row = [float(x) for x in line.strip().split(",")]
                    valid_X.append(row[1:self.input_dim + 1])
                    valid_X.append(row[self.input_dim + 1:-1])
                    y = int(row[-1])
                    if y == 0:
                        valid_y.append(0)
                        valid_y.append(1)
                    else:
                        valid_y.append(1)
                        valid_y.append(0)
                    valid_weights.append(row[0])
                    valid_weights.append(row[0])
                    valid_q.append(2)
        valid_X = np.array(valid_X)
        valid_y = np.array(valid_y)
        valid_q = np.array(valid_q)
        valid_weights = np.array(valid_weights)

        # Separate training data into components
        # train_weights = train_data[:, 0]  # target
        # train_y = train_data[:, -1].astype(int)
        # train_X1 = train_data[:, 1:self.input_dim + 1]
        # train_X2 = train_data[:, self.input_dim + 1:-1]
        # train_X = train_data[:, 1:-1]

        # Fit scaler using training data
        # # train_X = np.vstack([train_X1, train_X2])
        # self.scaler = StandardScaler()
        # self.scaler.fit(train_X)
        # joblib.dump(self.scaler, model_file + ".scaler")

        # Separate validation data into components
        # valid_weights = valid_data[:, 0]  # target
        # valid_y = valid_data[:, -1].astype(int)
        # valid_X1 = valid_data[:, 1:self.input_dim + 1]
        # valid_X2 = valid_data[:, self.input_dim + 1:-1]
        # valid_X = valid_data[:, 1:-1]

        # Scale training and validation data
        # train_X1 = self.scaler.transform(train_X1)
        # train_X2 = self.scaler.transform(train_X2)
        # valid_X1 = self.scaler.transform(valid_X1)
        # valid_X2 = self.scaler.transform(valid_X2)
        # train_X = self.scaler.transform(train_X)
        # valid_X = self.scaler.transform(valid_X)

        print("Train on {}, test on {}".format(train_X.shape, valid_X.shape))

        # Train model
        self.model = lgb.LGBMRanker()
        self.model.fit(X=train_X, y=train_y, sample_weight=train_weights,
                       group=train_q, eval_set=[(valid_X, valid_y)],
                       eval_group=[valid_q], eval_sample_weight=[
                           valid_weights],
                       eval_at=[1, 3], early_stopping_rounds=20, verbose=True,
                       callbacks=[lgb.reset_parameter(learning_rate=lambda x: 0.95 ** x * 0.1)])

        model_file = self.model_file.rstrip(
            ".h5") + "-" + str(self.iters) + ".h5"
        joblib.dump(self.model, model_file)

        self.prev_model = model_file
        self.load_model()

    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
        try:
            if self.prev_model is None or self.prev_model == "":
                self.prev_model = self.model_file

            if self.model is None or self.scaler is None:
                # self.build_model()
                self.load_model()

            # X1 = np.array(X1).reshape((1, self.input_dim))
            # X1 = self.scaler.transform(X1)
            # X2 = np.array(X2).reshape((1, self.input_dim))
            # X2 = self.scaler.transform(X2)
            # X = np.hstack((X1, X2))
            X = np.vstack((X1, X2))
            # X = self.scaler.transform(np.array([X]))

            pred = self.model.predict(X)
            # print(pred)
            # return pred
            # return int(pred)

            if (pred[0] >= pred[1]):
                return 1
            else:
                return 0
        except Exception as e:
            print('ERROR]: ' + str(e))
            return np.random.choice([0, 1])


# train_dirs = ["data/hybrid_bids/bids_620/train/iter1"]
# valid_dirs = ["data/hybrid_bids/bids_620/valid/iter1"]

# m = RankNet("models/hybrid_bids_620_lgbm.h5", 26, "")
# m.train_batch(train_dirs, valid_dirs, 100, 32)
# print(m.predict([1] * 26, [0] * 26))


# train_path = "data/hybrid_bids/bids_500/train/oracle"
# valid_path = "data/hybrid_bids/bids_500/valid/oracle"
# m = RankNet("models/hybrid_bids_500_lgbm-0.h5", 26, "models/hybrid_bids_500_lgbm-0.h5")

# print("TRAIN:")
# for file in Path(valid_path).glob("*.data"):
#     data = np.genfromtxt(file, delimiter=",", skip_header=1)
#     if data.ndim == 1:
#         continue
#     weights = data[:, 0]  # target
#     y = data[:, -1]
#     X = data[:, 1:-1]
#     y = y.reshape((len(y), 1))

#     s1 = m.model.predict(X)
#     s1 = s1.reshape((len(s1), 1))
#     s1 = (s1 >= 0.0).astype(int)
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
