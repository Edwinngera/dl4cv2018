from dlvc.dataset import Subset
from dlvc.datasets.pets import PetsDataset
from dlvc.test import Accuracy
from dlvc.models import pytorch as cnn
from dlvc import ops, batches

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

def load_dataset(subset: Subset, augment=False) -> batches.BatchGenerator:
    dataset = PetsDataset('../data/cifar-10-batches-py', subset)

    ops_list = []

    if augment:
        ops_list += [
            ops.hflip(),
            ops.rcrop(32, 12, 'constant')
        ]

    ops_list += [
        ops.mul(1 / 255),
        ops.type_cast(np.float32),
        # Imagenet:
        # ops.normalize(  mean=np.array([0.485, 0.456, 0.406]),
        #                 std=np.array([0.229, 0.224, 0.225])),
        # Cifar-10:
        ops.normalize(  mean=np.array([0.41477802, 0.45935813, 0.49693552]),
                        std=np.array([0.25241926, 0.24699265, 0.25279155])),
        ops.hwc2chw()
    ]

    op = ops.chain(ops_list)

    return batches.BatchGenerator(dataset, 128, True, op)

class PretrainedVGG19BnNet(nn.Module):
    def __init__(self):
        super(PretrainedVGG19BnNet, self).__init__()
        
        model = models.vgg19_bn(pretrained=True)

        self.features = model.features

        classifier_layers = list(model.classifier.children())[1:-1]
        classifier_layers = [nn.Linear(512, 4096, bias=True)] + classifier_layers + [nn.Linear(4096, 2, bias=True)]
        self.classifier = nn.Sequential(*classifier_layers)
    

    def enable_grad(self, enable):
        for param in self.features.parameters():
            param.requires_grad = enable

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    best_model_path = 'best_model.pth'

    training_batch = load_dataset(Subset.TRAINING, augment=True)
    validation_batch = load_dataset(Subset.VALIDATION)

    model = PretrainedVGG19BnNet()
    model.enable_grad(False)  # fix pre-trained layer weights

    if torch.cuda.is_available():
        model = model.cuda()

    learning_rate = 0.01
    weight_decay = 0.001

    cnn_cl = cnn.CnnClassifier(model, (3, 32, 32), num_classes=2, lr=learning_rate, wd=weight_decay, adam=False)

    loss_list = []
    measure = Accuracy()
    accuracies = []
    mean_loss_list = []

    best_accuracy = 0
    for epoch in range(1, 301):
        if epoch == 75:
            model.enable_grad(True) # enable fine-tuning of pre-trained layers
        
        if epoch == 150:
            for param_group in cnn_cl.optimizer.param_groups:
                param_group['lr'] = 0.001

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
    