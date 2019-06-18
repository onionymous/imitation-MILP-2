import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Add, Dense, Input, Lambda, Dropout, Subtract
from keras.models import Model, load_model
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

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

        self.is_gpu = is_gpu
        self.set_session()

        if prev_model and (prev_model != ""):
            self.build_model()
            self.load_model()

    def load_model(self):
        assert(self.prev_model and Path(self.prev_model).is_file()
               ), "Model {} could not be loaded.".format(self.prev_model)
        self.model.load_weights(self.prev_model)
        self.scaler = joblib.load(self.prev_model + ".scaler")

    def build_model(self):
        '''
        Setup the RankNet model architecture,
        '''

        h1_dim = 64
        h2_dim = h1_dim // 2
        h3_dim = h2_dim // 2

        # Model
        h1 = Dense(h1_dim, activation="relu")
        d1 = Dropout(rate=0.2)
        h2 = Dense(h2_dim, activation="relu")
        d2 = Dropout(rate=0.2)
        h3 = Dense(h3_dim, activation="relu")
        d3 = Dropout(rate=0.2)
        s = Dense(1)

        # Relevant samples
        rel = Input(shape=(self.input_dim, ), dtype="float32")
        h1_rel = h1(rel)
        d1_rel = d1(h1_rel)
        h2_rel = h2(d1_rel)
        d2_rel = d2(h2_rel)
        h3_rel = h3(d2_rel)
        d3_rel = d3(h3_rel)
        rel_score = s(d3_rel)

        # Irrelevant samples
        irr = Input(shape=(self.input_dim, ), dtype="float32")
        h1_irr = h1(irr)
        d1_irr = d1(h1_irr)
        h2_irr = h2(d1_irr)
        d2_irr = d2(h2_irr)
        h3_irr = h3(d2_irr)
        d3_irr = d3(h3_irr)
        irr_score = s(d3_irr)

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        self.model = Model(inputs=[rel, irr], outputs=prob)
        self.model.compile(optimizer="adagrad", loss="binary_crossentropy",
                           metrics=["acc"])

        self.scaler = StandardScaler()

    def set_session(self):
        '''
        Set the current session to either CPU or GPU depending on the value of
        the is_gpu flag of the class. This function assumes that no existing
        session exists (either not created or cleared with K.clear_session() )
        '''
        num_CPU = 1
        num_GPU = 0
        if self.is_gpu:
            num_GPU = 1

        config = tf.ConfigProto(intra_op_parallelism_threads=NUM_CORES,
                                inter_op_parallelism_threads=NUM_CORES,
                                allow_soft_placement=True,
                                device_count={'CPU': num_CPU,
                                              'GPU': num_GPU}
                                )

        session = tf.Session(config=config)
        K.set_session(session)

    def switch_session(self):
        '''
        Saves the model, clears the existing backend session, flips the is_gpu
        flag and creates a new session.
        '''
        # self.model.save(self.model_file)
        K.clear_session()
        self.is_gpu = not self.is_gpu
        self.set_session()

        self.model = None
        self.scaler = None

    def train(self, train_dir, valid_dir, num_epochs, batch_size):
        '''
        Run the training loop for a specified number of epochs and iterations.
        Training is done on the GPU so session will be switched if it was on
        CPU mode.
        '''

        # run training on the GPU
        if not self.is_gpu:
            self.switch_session()

        self.build_model()

        # Load data
        train_files = Path(train_dir).glob("*.data")
        train_data = np.concatenate(
            [np.genfromtxt(file, delimiter=",", skip_header=1) for file in train_files])

        valid_files = Path(valid_dir).glob("*.data")
        valid_data = np.concatenate(
            [np.genfromtxt(file, delimiter=",", skip_header=1) for file in valid_files])

        # Separate training data into components
        train_weights = train_data[:, 0]  # target
        train_y = train_data[:, -1]
        train_X1 = train_data[:, 1:self.input_dim + 1]
        train_X2 = train_data[:, self.input_dim + 1:-1]

        # Fit scaler using training data
        train_X = np.vstack([train_X1, train_X2])
        self.scaler.fit(train_X)
        joblib.dump(self.scaler, self.model_file + ".scaler")

        # Separate validation data into components
        valid_weights = valid_data[:, 0]  # target
        valid_y = valid_data[:, -1]
        valid_X1 = valid_data[:, 1:self.input_dim + 1]
        valid_X2 = valid_data[:, self.input_dim + 1:-1]

        # Scale training and validation data
        train_X1 = self.scaler.transform(train_X1)
        train_X2 = self.scaler.transform(train_X2)
        valid_X1 = self.scaler.transform(valid_X1)
        valid_X2 = self.scaler.transform(valid_X2)

        # Train model
        checkpointer = ModelCheckpoint(
            filepath=self.model_file, verbose=2, save_best_only=True)
        history = self.model.fit([train_X1, train_X2], train_y,
                                 sample_weight=train_weights,
                                 epochs=num_epochs, batch_size=batch_size,
                                 validation_data=(
                                     [valid_X1, valid_X2], valid_y, valid_weights),
                                 callbacks=[checkpointer], verbose=2)

        # print("avg prediction: ", np.mean(
        #    self.model.predict([train_X1, train_X2])))

        self.prev_model = self.model_file
        self.load_model()

        print(self.model.evaluate(
            [train_X1, train_X2], train_y, batch_size=batch_size, verbose=2))
        print(self.model.evaluate(
            [valid_X1, valid_X2], valid_y, batch_size=batch_size, verbose=2))

    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
        # predict on the CPU
        if self.is_gpu:
            self.switch_session()

        if self.prev_model is None or self.prev_model == "":
            self.prev_model = self.model_file

        if self.model is None or self.scaler is None:
            self.build_model()
            self.load_model()

        X1 = np.array(X1).reshape((1, self.input_dim))
        X1 = self.scaler.transform(X1)
        X2 = np.array(X2).reshape((1, self.input_dim))
        X2 = self.scaler.transform(X2)

        pred = np.asscalar(self.model.predict([X1, X2]))

        if (pred >= 0.5):
            return 1
        else:
            return 0


# train_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_train/data/"
# valid_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_valid/data/"
m = RankNet("models/mvc3.h5", 26, "")
# m.train(train_path, valid_path, 100, 32)
# print(m.predict([0] * 26, [1] * 26))

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
