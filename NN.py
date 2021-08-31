import multiprocessing

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.utils.data as data

import torchvision
from torch import optim, nn
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 10
learning_rate = 0.003
num_epochs = 6

train_data_path = "images/fruits/Training"
test_data_path = "images/fruits/Test"

classes = ["Apple", "NotApple"]

"""
================================================================
                        Show Images
================================================================
"""


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""
================================================================
                        Processing data
================================================================
"""

transform_img = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

train_set = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform_img)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True, num_workers=2)

test_set = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_img)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=True, num_workers=2)


def NN():
    # show images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    """
    ================================================================
                            Define model
    ================================================================
    """

    model = torchvision.models.resnet34(pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    """
    ================================================================
                            Train Model
    ================================================================
    """
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0

        # training-the-model
        model.train()
        for data, target in train_loader:
            # move-tensors-to-GPU
            data = data.to(device)
            target = target.to(device)

            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # calculate-the-batch-loss
            loss = criterion(output, target)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)

        # calculate-average-losses
        train_loss = train_loss / len(train_loader.sampler)
        train_losses.append(train_loss)

        # print-training/validation-statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    # Save
    torch.save(model.state_dict(), 'model.pth')

    """
    ================================================================
                            Evaluate Model
    ================================================================
    """


def eval():
    model = torchvision.models.resnet34()
    model.to(device)
    model.load_state_dict(torch.load('model.pth'))

    # test-the-model
    model.eval()  # it-disables-dropout
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('\n\nCorrect: {}\t total: {}'.format(correct, total))
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    eval()
