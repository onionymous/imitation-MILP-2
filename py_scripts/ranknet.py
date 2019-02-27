import numpy as np
from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Add, Dense, Input, Lambda
from keras.models import Model
from pathlib import Path


class RankNet:

    def __init__(self, model_file, input_dim, prev_model=None):
        self.model_file = model_file
        self.input_dim = input_dim

        h_1_dim = 64
        h_2_dim = h_1_dim // 2
        h_3_dim = h_2_dim // 2

        # Model
        h_1 = Dense(h_1_dim, activation="relu")
        h_2 = Dense(h_2_dim, activation="relu")
        h_3 = Dense(h_3_dim, activation="relu")
        s = Dense(1)

        # Relevant samples
        rel = Input(shape=(self.input_dim, ), dtype="float32")
        h_1_rel = h_1(rel)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # Irrelevant samples
        irr = Input(shape=(self.input_dim, ), dtype="float32")
        h_1_irr = h_1(irr)
        h_2_irr = h_2(h_1_irr)
        h_3_irr = h_3(h_2_irr)
        irr_score = s(h_3_irr)

        # Subtract scores.
        negated_irr_score = Lambda(
            lambda x: -1 * x, output_shape=(1, ))(irr_score)
        diff = Add()([rel_score, negated_irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        self.model = Model(inputs=[rel, irr], outputs=prob)
        self.model.compile(optimizer="adagrad", loss="binary_crossentropy",
                           metrics=["acc"])

        # Load weights if it was specified
        if prev_model and Path(prev_model).is_file():
            self.model.load_weights(prev_model)

        # Get score function
        self.get_score = backend.function([rel], [rel_score])

    def train(self, train_file, valid_file, num_epochs, batch_size):
        # y = np.ones((train_X.shape[0], 1))
        train_data = np.genfromtxt(train_file, delimiter=",", skip_header=1)
        valid_data = np.genfromtxt(valid_file, delimiter=",", skip_header=1)

        train_weights = train_data[:, 0]  # target
        train_y = train_data[:, -1]
        train_X1 = train_data[:, 1:self.input_dim + 1]
        train_X2 = train_data[:, self.input_dim + 1:-1]

        valid_weights = valid_data[:, 0]  # target
        valid_y = valid_data[:, -1]
        valid_X1 = valid_data[:, 1:self.input_dim + 1]
        valid_X2 = valid_data[:, self.input_dim + 1:-1]

        # Train model
        checkpointer = ModelCheckpoint(
            filepath=self.model_file, verbose=1, save_best_only=True)
        history = self.model.fit([train_X1, train_X2], train_y,
                                 sample_weight=train_weights,
                                 epochs=num_epochs, batch_size=batch_size,
                                 validation_data=(
                                     [valid_X1, valid_X2], valid_y, valid_weights),
                                 callbacks=[checkpointer], verbose=2)

        print("avg prediction: ", np.mean(
            self.model.predict([train_X1, train_X2])))

    def predict(self, X1, X2):
        X1 = np.array(X1).reshape((1, self.input_dim))
        X2 = np.array(X2).reshape((1, self.input_dim))

        s1 = np.asscalar(self.model.predict([X1, X2]))
        s2 = np.asscalar(self.model.predict([X2, X1]))

        if (s1 > s2):
            return 1
        else:
            return 0


# m = RankNet("models/test2.h5", 26, "models/test.h5")
# m.train("test_data/test_data.train",
#         "test_data/test_data.valid", 10, 32)
# m.predict(np.array([1] * 26), np.array([0] * 26))
