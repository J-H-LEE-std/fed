import warnings

from collections import OrderedDict

import flwr as fl
from torch.multiprocessing import Process
import numpy as np
import json
import random
import math

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import SGD

warnings.filterwarnings("ignore", category=UserWarning)

class FemnistDataset(Sequence):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.x = [tf.convert_to_tensor(np.array(i).reshape(28,28,1)) for i in dataset['x']]
        self.y = [tf.convert_to_tensor(np.array(j).reshape(28,28,1)) for j in dataset['y']]
        self.batch_size = batch_size
        self.shuffle=shuffle

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

class femnist_network(Model):
    def __init__(self):
        super(femnist_network, self).__init__()
        self.input = layers.Input(shape=(28, 28, 1))
        self.zero1 = layers.ZeroPadding2D(padding=(2,2))
        self.conv1 = layers.Conv2d(filters=32, kernel_size=(5,5), padding='valid', activation='relu')
        self.maxpool1 = layers.MaxPooling2d((2, 2))
        self.zero2 = layers.ZeroPadding2D(padding=(2,2))
        self.conv2 = layers.Conv2d(filters=64, kernel_size=(5,5), padding='valid', activation='relu')
        self.maxpool2 = layers.MaxPooling2d((2, 2))
        self.flat = layers.Flatten()
        self.linear1 = layers.Dense(2048, activation='relu')
        self.linear2 = layers.Dense(62, activation='softmax')

    def forward(self, x):
        x = self.input(x)
        x = self.conv1(self.zero1(x))
        x = self.maxpool1(x)
        x = self.conv2(self.zero2(x))
        x = self.maxpool2(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def main():
    number = random.randint(0, 35)
    if number == 35:
        subject_number = random.randint(0, 96)
    else:
        subject_number = random.randint(0, 99)
    print('number : {}, subject number : {}'.format(number, subject_number))
    with open("./data/data/train/all_data_"+str(number)+"_niid_0_keep_0_train_9.json","r") as f:
        train_json = json.load(f)
    with open("./data/data/test/all_data_"+str(number)+"_niid_0_keep_0_test_9.json","r") as f:
        test_json = json.load(f)
    train_user = train_json['users'][subject_number]
    train_data = train_json['user_data'][train_user]
    test_user = test_json['users'][subject_number]
    test_data = test_json['user_data'][test_user]

    trainset = FemnistDataset(train_data, batch_size=64, shuffle=True)
    testset = FemnistDataset(test_data, batch_size=64)

    model = femnist_network()
    optimizer = SGD(learning_rate=0.0003)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(trainset, epochs=20, batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(train_data['x']), {}

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(testset)
            return loss, len(test_data['x']), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client(server_address="127.0.0.1:8888", client=CifarClient())

if __name__ == "__main__":
    list = list(range(1, 10))

    ps = []
    for i in list:
        p =Process(target=main)
        ps.append(p)
        p.start()
    for p in ps:
        p.join()