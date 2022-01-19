import multiprocessing
import time

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.utils.data as data
import torchvision
from torch import optim, nn
from torchvision import transforms

# set CUDA device if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# set batch size
batch_size = 8
# set model learning rate and number of epochs
learning_rate = 0.003
num_epochs = 1

# set train and test directory
train_data_path = "images/fruits/Training"
test_data_path = "images/fruits/Test"

# set number of classes and their names
classes = ["Apple", "NotApple"]

# ================================================================
#                        Processing data
# ================================================================

transform_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
# set loader for training and testing datasets
train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform_img)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)
test_set = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_img)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True, num_workers=2)


# function to print timestamps
def timestamp():
    return print(time.strftime('[%H:%M:%S]\t...\t'), end=' ')


# function to show images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# function to train model
def train():
    timestamp(), print('Training begins ...\nPlease wait ...')
    # iterate through train dataset loader
    timestamp(), print('Loading data')
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # show images
    # timestamp(), print('Showing images')
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    # timestamp(), print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # ================================================================
    #                         Define model
    # ================================================================
    # choose NN model with pretrained weights
    model = torchvision.models.resnet34(pretrained=True)
    timestamp(), print(f'Setting model : \n{model}')
    # perform on the CUDA supported GPU
    model.to(device)
    timestamp(), print(f'Migrating to : {device}')
    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    timestamp(), print(f'Setting criterion : {criterion}')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    timestamp(), print(f'Setting optimizer : {optimizer}')

    # ================================================================
    #                        Train model
    # ================================================================
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        timestamp(), print(f'[{epoch}/{num_epochs}] epoch')
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # training the model
        model.train()
        for data, target in train_loader:
            # move tensors to CUDA supported GPU
            data = data.to(device)
            target = target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward-pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward-pass: compute gradient of the loss wrt model parameters
            loss.backward()
            # perform a single optimization step (parameter-update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        train_losses.append(train_loss)
        # print training/validation statistics
        print('Train Loss: {:.6f}'.format(train_loss))

    # Save trained NN weights to model.pth
    timestamp(), print(f'Saving model')
    torch.save(model.state_dict(), 'model.pth')


# ================================================================
#                        Evaluate Model
# ================================================================


def evaluate():
    timestamp(), print(f'Evaluating model\nPlease,wait ...')
    # create a NN, which model is similar to which is used during training
    model = torchvision.models.resnet34()
    timestamp(), print(f'Setting model : \n{model}')
    # perform on CUDA supported GPU if available
    model.to(device)
    timestamp(), print(f'Migrating to : {device}')
    # load weights of trained NN
    timestamp(), print(f'Loading weights')
    model.load_state_dict(torch.load('model.pth'))

    # test the model
    timestamp(), print(f'Entering evaluation mode')
    model.eval()  # evaluation mode
    timestamp(), print(f'Disabling gradient calculation')
    with torch.no_grad():  # disable gradient calculation
        correct = 0
        total = 0

        # iterate through data in test loader
        for images, labels in test_loader:
            # perform on CUDA supported GPU if available
            images = images.to(device)
            labels = labels.to(device)
            # pass test images through NN
            outputs = model(images)
            # find predictions
            _, predicted = torch.max(outputs.data, 1)
            # keep track of total images passed through NN
            total += labels.size(0)
            # keep track of correct predictions
            correct += (predicted == labels).sum().item()

        # print results in console
        timestamp(), print('Results:\n\t\t\t\t\tCorrect: {}\t total: {}'.format(correct, total))
        print('\t\t\t\t\tTest Accuracy of the model: {:.2f} %'.format(100 * correct / total))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # train()
    evaluate()
