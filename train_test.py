import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import os
import time
from DenseNet import *

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=9, type=int)
parser.add_argument('--batchsize', default=10, type=int)
parser.add_argument('--learningrate', default=0.01, type=float)
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    os.chdir('./datasets')
    download = False if os.path.isfile('train_32x32.mat') else True
    os.chdir('../')
    trainset = torchvision.datasets.SVHN(root='./datasets', split='train', download=download, transform=transforms.ToTensor())

    # Hyperparameters
    batchsize = args.batchsize
    momentum = 0.9
    lr = args.learningrate
    nr_classes = 10
    depth = 100
    nr_epochs = args.epochs

    # Load the model on the GPU
    model = DenseNet(depth , nr_classes)
    model.cuda()
    
    # Data Loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=4)

    epoch_no = []
    batch_no = []
    loss_value = []
    time_elapsed = []
    loss_vctr = []
    counter = 0
    total = nr_epochs * int((73257 / batchsize) / 2000)

    # Oprimization Criteria and Optimization Method
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)

    # Training Loop
    start = time.time()
    print('\nTraining started\n')
    for epoch in range(nr_epochs):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda() 
            labels = labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass + backpropagation + optimization
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # save statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                counter += 1
                print('%.2f%% complete' % (100 * counter / total))
                loss_vctr.append(running_loss / 2000)
                end = time.time()
                epoch_no.append(epoch + 1)
                batch_no.append(i + 1)
                loss_value.append(round((running_loss / 2000), 3))
                time_elapsed.append(round(((end - start) / 60), 2))
                running_loss = 0.0    
    print('\nTraining finished\n')

    # Save training data in a csv file
    dict_train = {'Epoch Number': epoch_no, 'Mini-Batch number': batch_no, 'Loss': loss_value, 'Time elapsed (minutes)': time_elapsed}
    df_train = pd.DataFrame(dict_train)
    df_train.to_csv('./misc/training_stats.csv')

    # Plot training loss
    x = range(1, nr_epochs * int((73257 / batchsize) / 2000) + 1)
    x_epoch = [z for z in range(1, len(x) + 1) if z % ((73257 // batchsize) // 2000) == 0] 
    x_ticks_labels = ['epoch ' + str(y) for y in range(1, nr_epochs+1)]
    plt.figure(figsize=(14,7))
    plt.plot(x , loss_vctr)
    plt.xticks(x_epoch, x_ticks_labels)
    plt.title('Training Loss')
    plt.savefig('./img/training_loss.png')

    # Testing
    os.chdir('./datasets')
    download = False if os.path.isfile('test_32x32.mat') else True
    os.chdir('../')
    testset = torchvision.datasets.SVHN(root='./datasets', split='test', download=download, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=4)

    confusion_matrix = torch.zeros(nr_classes, nr_classes)
    print('\nTesting started')
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.cuda().data, 1)
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
    print('Testing finished\n')

    precision = []
    recall = []
    accuracy = []
    for i in range(nr_classes):
        accuracy.append(round(int(confusion_matrix.diag().sum()) / int(confusion_matrix.sum()),2))
        precision.append(round(int(confusion_matrix[i,i]) / int(confusion_matrix.sum(1)[i]),2))
        recall.append(round(int(confusion_matrix[i,i]) / int(confusion_matrix.sum(0)[i]),2))

    dict_test = {'Label': [z for z in range(nr_classes)], 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
    df_test = pd.DataFrame(dict_test)
    df_test.to_csv('./misc/performance_metrics.csv')

    # Confusion matrix heatmap
    cm = confusion_matrix.numpy()
    plt.figure(figsize=(12,7))
    hm = sns.heatmap(data=cm, cmap='Blues', annot=True, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Actual label')
    plt.ylabel('Predicted label')
    plt.savefig('./img/confusion_matrix.png')