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

NUM_CORES = 4


def count_data(file_path):
    '''
    Fast count of number of lines in a file. Total data points is number of
    lines minus one for the header.
    '''
    return sum(1 for i in open(file_path, 'rb')) - 1


class DataGenerator():
    '''
    Directory-based data generator for Keras models.
    '''

    def __init__(self, data_dir, dim, batch_size=32, shuffle=True, is_loop=True):
        self.data_dir = data_dir
        self.files = []
        self.n_samples = 0
        self.n_batches = 0

        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_loop = is_loop

        self.next_file_index = 0
        self.data_index = 0

        # Get all the data files and their sizes
        for file in Path(self.data_dir).glob("*.data"):
            n = count_data(file)
            self.files.append(str(file))
            self.n_samples += n
            self.n_batches += int(np.ceil(n / self.batch_size))

        if self.n_samples == 0:
            raise ValueError("Number of total samples is zero")

        # Shuffle the data and initialize
        np.random.shuffle(self.files)
        self.load_next_file()

    def steps_per_epoch(self):
        '''
        Number of batches per epoch.
        '''
        # if self.n_samples < self.batch_size:
        #     return int(np.ceil(self.n_samples / self.batch_size))
        # else:
        #     return int(np.floor(self.n_samples / self.batch_size))
        return self.n_batches

    def load_next_file(self):
        '''
        Load the next file.
        '''
        # Loop around to beginning
        if self.next_file_index == len(self.files):
            self.next_file_index = 0
            np.random.shuffle(self.files)

        # Read next file into memory
        self.curr_data = np.genfromtxt(
            self.files[self.next_file_index], delimiter=",", skip_header=1)

        # Reset data index
        self.next_file_index += 1
        self.data_index = 0

    def get_batch(self):
        '''
        Generate a batch of data.
        '''
        # First get the data
        data = np.empty((0, 2 * (self.dim + 1)))

        # Read a batch of data, looping back to the beginning of the files if
        # necessary
        # n_needed = self.batch_size
        end = min(len(self.curr_data), self.data_index + self.batch_size)
        data = np.vstack((data, self.curr_data[self.data_index:end, ]))
        if (end == len(self.curr_data)):
            self.load_next_file
        else:
            self.data_index += self.batch_size
        # while len(data) < self.batch_size:
        #     self.load_next_file()

        #     # Read whatever extra lines needed
        #     n_needed = self.batch_size - len(data)
        #     end = min(len(self.curr_data), self.data_index + n_needed)
        #     data = np.vstack((data, self.curr_data[self.data_index:end, ]))
        # self.data_index += n_needed

        # Split up the data accordingly
        weights = data[:, 0]  # target
        y = data[:, -1]
        X1 = data[:, 1:self.dim + 1]
        X2 = data[:, self.dim + 1:-1]

        return ([X1, X2], y, weights)


def build_standard_scaler(data_path, dim):
    '''
    Build a scaler based on the current training data
    '''
    scaler = StandardScaler()
    # for file in Path(data_path).glob("*.data"):
    #     data = np.genfromtxt(file, delimiter=",", skip_header=1)[:, 1:-1]
    #     data = np.vstack((data[:, :dim], data[:, dim:]))
    #     scaler.partial_fit(data)
    return scaler


def data_generator(data_path, scaler, input_dim, batch_size):
    '''
    Iterator wrapper around the DataGenerator class that yields batches.
    '''
    dg = DataGenerator(data_path, input_dim,
                       batch_size=batch_size, shuffle=True)
    idx = 0
    while True:
        # (xs, y, weights) = dg.get_batch()
        # X1 = scaler.transform(xs[0])
        # X2 = scaler.transform(xs[1])
        data = np.genfromtxt(dg.files[idx], delimiter=",", skip_header=1)
        weights = data[:, 0]  # target
        y = data[:, -1]
        X1 = data[:, 1:dg.dim + 1]
        X2 = data[:, dg.dim + 1:-1]

        for i in range(0, data.shape[0], batch_size):
            end = min(i + batch_size, data.shape[0])
            X1_batch = X1[i:end]
            X2_batch = X2[i:end]
            y_batch = y[i:end]
            w_batch = weights[i:end]
            yield ([X1_batch, X2_batch], y_batch, w_batch)

        idx += 1
        if idx == len(dg.files):
            idx = 0
            np.random.shuffle(dg.files)
        # yield (xs, y, weights)


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

        self.is_gpu = is_gpu

        self.set_session()
        if prev_model and (prev_model != ""):
            self.build_model()
            self.load_model()

    def load_model(self):
        assert(self.prev_model and Path(self.prev_model).is_file()
               ), "Previous model {} could not be loaded.".format(self.prev_model)
        self.model.load_weights(self.prev_model)

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

        # Create data scaler
        scaler = build_standard_scaler(train_dir, self.input_dim)

        # Create generators
        train_dg = DataGenerator(
            train_dir, self.input_dim, batch_size=batch_size, shuffle=True)
        valid_dg = DataGenerator(
            valid_dir, self.input_dim, batch_size=batch_size, shuffle=True)

        train_gen = data_generator(
            train_dir, scaler, self.input_dim, batch_size)
        valid_gen = data_generator(
            valid_dir, scaler, self.input_dim, batch_size)

        # Count data files
        print("Train on {} samples, test on {} samples".format(
            train_dg.n_samples, valid_dg.n_samples))

        # Train model
        checkpointer = ModelCheckpoint(
            filepath=self.model_file, verbose=4, save_best_only=True)

        # for file in Path(self.data_dir).glob("*.data"):
        #     data = np.genfromtxt(file, delimiter=",", skip_header=1)

        #     weights = data[:, 0]  # target
        #     y = data[:, -1]
        #     X1 = data[:, 1:self.dim + 1]
        #     X2 = data[:, self.dim + 1:-1]

        #     history = self.model.train_on_batch([X1, X2], y, weights)

        history = self.model.fit_generator(train_gen,
                                           steps_per_epoch=train_dg.steps_per_epoch(),
                                           validation_data=valid_gen,
                                           validation_steps=valid_dg.steps_per_epoch(),
                                           epochs=num_epochs,
                                           callbacks=[checkpointer], verbose=1,
                                           max_queue_size=10, workers=1, use_multiprocessing=False,
                                           shuffle=False)

        self.prev_model = self.model_file
        self.load_model()

        print(self.model.evaluate_generator(train_gen, train_dg.steps_per_epoch(),
                                            max_queue_size=10, workers=1, use_multiprocessing=False))
        print(self.model.evaluate_generator(valid_gen, train_dg.steps_per_epoch(),
                                            max_queue_size=10, workers=1, use_multiprocessing=False))

    def predict(self, X1, X2):
        '''
        Get a prediction on which node is better given the two feature vectors
        of the nodes.
        '''
        # predict on the CPU
        if self.is_gpu:
            self.switch_session()
            self.build_model()
            self.load_model()

        X1 = np.array(X1).reshape((1, self.input_dim))
        X2 = np.array(X2).reshape((1, self.input_dim))

        pred = np.asscalar(self.model.predict([X1, X2]))

        if (pred >= 0.5):
            return 1
        else:
            return 0


train_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_train/data"
valid_path = "/home/orion/Documents/dev/imitation-milp-2/data/3dp_valid/data/"
m = RankNet("models/3dp3.h5", 26, "models/3dp3.h5")
# m.train(train_path, valid_path, 10, 32)
# m.predict([0] * 26, [1] * 26)

print("TRAIN:")
for file in sorted(Path(train_path).glob("*.data")):
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    weights = data[:, 0]  # target
    y = data[:, -1]
    X1 = data[:, 1:27]
    X2 = data[:, 27:-1]

    print(file)
    print(m.model.evaluate([X1, X2], y, batch_size=32, verbose=2))

print("\nVALID:")
for file in sorted(Path(valid_path).glob("*.data")):
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    weights = data[:, 0]  # target
    y = data[:, -1]
    X1 = data[:, 1:27]
    X2 = data[:, 27:-1]

    print(file)
    print(m.model.evaluate([X1, X2], y, batch_size=32, verbose=2))
