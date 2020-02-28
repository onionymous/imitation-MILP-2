import sys
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score

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
        self.scaler = joblib.load(self.prev_model + ".scaler")

    def train(self, train_dir, valid_dir, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        def get_predicted_outcome(model, data):
            return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)

        def get_predicted_rank(model, data):
            return model.predict_proba(data)[:, 1]

        def train_model(model, prediction_function, X_train, y_train, w_train, X_test, y_test):
            model.fit(X_train, y_train, sample_weight=w_train)

            y_train_pred = prediction_function(model, X_train)
            print('train precision: ' +
                  str(precision_score(y_train, y_train_pred)))
            print('train recall: ' + str(recall_score(y_train, y_train_pred)))
            print('train accuracy: ' + str(accuracy_score(y_train, y_train_pred)))
            y_test_pred = prediction_function(model, X_test)
            print('test precision: ' + str(precision_score(y_test, y_test_pred)))
            print('test recall: ' + str(recall_score(y_test, y_test_pred)))
            print('test accuracy: ' + str(accuracy_score(y_test, y_test_pred)))

            return model

        # Load data
        train_data = np.empty(2 * self.input_dim + 2)
        # for train_dir in train_dirs:
        train_files = Path(train_dir).glob("*.data")
        for file in train_files:
            data = np.genfromtxt(file, delimiter=",", skip_header=1)
            if data.ndim == 1:
                continue
            train_data = np.row_stack((train_data, data))

        valid_data = np.empty(2 * self.input_dim + 2)
        # for valid_dir in valid_dirs:
        valid_files = Path(valid_dir).glob("*.data")
        for file in valid_files:
            data = np.genfromtxt(file, delimiter=",", skip_header=1)
            if data.ndim == 1:
                continue
            valid_data = np.row_stack((valid_data, data))

        # Separate training data into components
        train_weights = train_data[:, 0]  # target
        train_y = train_data[:, -1].astype(int)
        # train_X1 = train_data[:, 1:self.input_dim + 1]
        # train_X2 = train_data[:, self.input_dim + 1:-1]
        train_X = train_data[:, 1:-1]

        # Fit scaler using training data
        model_file = self.model_file.rstrip(
            ".h5") + "-" + str(self.iters) + ".h5"
        # train_X = np.vstack([train_X1, train_X2])
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        joblib.dump(self.scaler, model_file + ".scaler")

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
        train_X = self.scaler.transform(train_X)
        valid_X = self.scaler.transform(valid_X)

        print("Train on {}, test on {}".format(train_X.shape, valid_X.shape))

        # Train model
        self.model = train_model(
            LogisticRegression(), get_predicted_outcome, train_X, train_y, train_weights, valid_X, valid_y)

        joblib.dump(self.model, model_file)

        self.prev_model = model_file
        self.load_model()

    def train_batch(self, train_dirs, valid_dirs, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        def get_predicted_outcome(model, data):
            return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)

        def get_predicted_rank(model, data):
            return model.predict_proba(data)[:, 1]

        def train_model(model, prediction_function, X_train, y_train, w_train, X_test, y_test):
            model.fit(X_train, y_train, sample_weight=w_train)

            y_train_pred = prediction_function(model, X_train)
            print('train precision: ' +
                  str(precision_score(y_train, y_train_pred)))
            print('train recall: ' + str(recall_score(y_train, y_train_pred)))
            print('train accuracy: ' + str(accuracy_score(y_train, y_train_pred)))
            y_test_pred = prediction_function(model, X_test)
            print('test precision: ' + str(precision_score(y_test, y_test_pred)))
            print('test recall: ' + str(recall_score(y_test, y_test_pred)))
            print('test accuracy: ' + str(accuracy_score(y_test, y_test_pred)))

            return model

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
        train_weights = train_data[:, 0]  # target
        train_y = train_data[:, -1].astype(int)
        # train_X1 = train_data[:, 1:self.input_dim + 1]
        # train_X2 = train_data[:, self.input_dim + 1:-1]
        train_X = train_data[:, 1:-1]

        # Fit scaler using training data
        model_file = self.model_file.rstrip(
            ".h5") + "-" + str(self.iters) + ".h5"
        # train_X = np.vstack([train_X1, train_X2])
        self.scaler = StandardScaler()
        self.scaler.fit(train_X)
        joblib.dump(self.scaler, model_file + ".scaler")

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
        train_X = self.scaler.transform(train_X)
        valid_X = self.scaler.transform(valid_X)

        print("Train on {}, test on {}".format(train_X.shape, valid_X.shape))

        # Train model
        self.model = train_model(
            LogisticRegression(), get_predicted_outcome, train_X, train_y, train_weights, valid_X, valid_y)

        joblib.dump(self.model, model_file)

        self.prev_model = model_file
        self.load_model()


    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
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
        X = np.hstack((X1, X2))
        X = self.scaler.transform(np.array([X]))

        pred = self.model.predict(X).item()
        return int(pred)

        # if (pred >= 0.5):
        #     return 1
        # else:
        #     return 0

# train_dirs = ["data/hybrid_bids/bids_500/train/oracle", "data/hybrid_bids/bids_530/train/scaling"]
# valid_dirs = ["data/hybrid_bids/bids_500/valid/oracle", "data/hybrid_bids/bids_530/valid/scaling"]

# m = RankNet("models/hybrid_bids_lr_scaling.h5", 26, "")
# m.train_batch(train_dirs, valid_dirs, 100, 32)
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
