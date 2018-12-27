from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.test import Accuracy
from dlvc.models import knn, pytorch as cnn
from dlvc import ops, batches

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

INPUT_DIM = 3072
NUM_CLASSES = 2


def load_dataset(subset: Subset, augment=False) -> batches.BatchGenerator:
    dataset = PetsDataset('../data/cifar-10-batches-py', subset)

    ops_list = []

    if augment:
        ops_list += [
            ops.hflip(),
            ops.resize((224, 224)),
            ops.rcrop(224, 32, 'reflect')
        ]

    ops_list += [
        ops.mul(1 / 255),
        ops.type_cast(np.float32),
        # Cifar-10:
        ops.normalize(  mean=np.array([0.41477802, 0.45935813, 0.49693552]),
                        std=np.array([0.25241926, 0.24699265, 0.25279155])),
        ops.hwc2chw()
    ]

    op = ops.chain(ops_list)

    return batches.BatchGenerator(dataset, 128, True, op)

import torchvision.models as models

def get_pretrained_resnet18_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(*[nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)])    
    return model

def get_pretrained_resnet50_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(*[nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)])    
    return model

def get_pretrained_vgg11bn_model():
    model = models.vgg11_bn(pretrained=True)
    classifier_layers = list(model.classifier.children())[:-1]
    classifier_layers = classifier_layers + [nn.Linear(4096, 2, bias=True)]
    model.classifier = nn.Sequential(*classifier_layers)
    return model

def get_pretrained_vgg16bn_model():
    model = models.vgg16_bn(pretrained=True)
    classifier_layers = list(model.classifier.children())[:-1]
    classifier_layers = classifier_layers + [nn.Linear(4096, 2, bias=True)]
    model.classifier = nn.Sequential(*classifier_layers)
    return model

def get_pretrained_model():
    # return get_pretrained_resnet18_model()
    return get_pretrained_resnet50_model()
    # return get_pretrained_vgg11bn_model()
    # return get_pretrained_vgg16bn_model()

def enable_grad(model, enable):
    for param in model.parameters():
        param.requires_grad = enable
    
    if model.classifier: # vgg
        for param in model.classifier.parameters():
            param.requires_grad = True

    if model.fc: # resnet
        for param in model.fc.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    best_model_path = 'best_model.pth'

    training_batch = load_dataset(Subset.TRAINING, augment=True)
    validation_batch = load_dataset(Subset.VALIDATION)

    model = get_pretrained_model()
    enable_grad(model, False)

    if torch.cuda.is_available():
        model = model.cuda()

    learning_rate = 0.001
    weight_decay = 0.001

    cnn_cl = cnn.CnnClassifier(model, (3, 224, 224), num_classes=2, lr=learning_rate, wd=weight_decay, adam=False)

    loss_list = []
    measure = Accuracy()
    accuracies = []
    mean_loss_list = []

    best_accuracy = 0
    for epoch in range(1, 201):
        if epoch == 50:
            enable_grad(model, True) # enable fine-tuning of pre-trained layers at epoch 50
        
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

        if measure.accuracy() > best_accuracy:
            best_accuracy = measure.accuracy()
            torch.save(model.state_dict(), best_model_path)
            print('\tNew best accuracy. Saved model to: "{}"'.format(best_model_path))

        accuracies.append(measure.accuracy())
    
    print('Best accuracy: {0:0.3f}'.format(best_accuracy))
    