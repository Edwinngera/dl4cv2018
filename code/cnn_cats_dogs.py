from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.test import Accuracy
from dlvc.models import pytorch as cnn
from dlvc import ops, batches

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_dataset(subset: Subset) -> batches.BatchGenerator:
    dataset = PetsDataset('../data/cifar-10-batches-py', subset)

    op = ops.chain([
        ops.hwc2chw(),
        ops.add(-127.5),
        ops.mul(1 / 127.5),
        ops.type_cast(np.float32)
    ])

    return batches.BatchGenerator(dataset, 128, True, op)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Deep conv filters produce better results
        # Adding more conv layers does not seem to help
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    training_batch = load_dataset(Subset.TRAINING)
    validation_batch = load_dataset(Subset.VALIDATION)

    net = Net()
    
    if torch.cuda.is_available():
        net = net.cuda()

    learning_rate = 0.001
    weight_decay = 0

    cnn_cl = cnn.CnnClassifier(net, (3, 32, 32), num_classes=2, lr=learning_rate, wd=weight_decay)

    loss_list = []
    measure = Accuracy()
    accuracies = []
    mean_loss_list = []

    best_accuracy = 0
    for epoch in range(1, 301):
        predictions = np.zeros((1, 2))
        loss_list = []
        labels = []
        for training_data in training_batch:
            loss = cnn_cl.train(training_data.data, training_data.label)
            loss_list.append(loss)

        mean_loss = np.array(loss_list).mean()
        std_loss = np.array(loss_list).std()
        print('epoch: {}'.format(epoch))
        print('\ttrain loss: {0:0.3f} +- {1:0.3f}'.format(mean_loss, std_loss))
        mean_loss_list.append(mean_loss)

        for validation_data in validation_batch:
            prediction = cnn_cl.predict(validation_data.data)

            predictions = np.vstack((predictions, prediction))
            labels.append(validation_data.label)

        predictions = predictions[1:]
        labels = np.array(labels)
        labels = np.hstack(labels)

        measure.reset()
        measure.update(predictions, labels)
        print('\tval acc: accuracy: {0:0.3f}'.format(measure.accuracy()))
        accuracies.append(measure.accuracy())

        if measure.accuracy() > best_accuracy:
            best_accuracy = measure.accuracy()
    
    print('Best accuracy: {0:0.3f}'.format(best_accuracy))