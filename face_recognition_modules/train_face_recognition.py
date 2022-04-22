from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
import time
import copy
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
from models.inception_resnet_v1 import InceptionResnetV1
import build_custom_model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.imsave("show_data.png",inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    FT_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
    # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                
                FT_losses.append(loss.item())
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    time_elapsed = time.time() - since
    print(f'===>Training for {num_epochs} epochs completes in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print('===>Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, FT_losses, best_acc



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train your customized face recognition model')
    parser.add_argument('--data_dir', type=str, default="./data/test_me", help='the path of the dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='training epochs')
    args = parser.parse_args()


    data_dir = args.data_dir
    num_epochs = args.num_epochs
    labels_dir = "./checkpoint/labels.json" # the path to save labels
    model_path = "./checkpoint/model_vggface2_best.pth" # the path to save best model weights


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
    class_names = image_datasets['train'].classes
    print(f"Class names: {class_names}")


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make a grid from batch
    out = utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    model_ft = build_custom_model.build_model(len(class_names))
    model_ft = model_ft.to(device)
    print('Running on device: {}'.format(device))


    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=1e-2, momentum=0.9)
    # Decay LR by a factor of *gamma* every *step_size* epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # train and evaluate
    model_ft, FT_losses, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)

    
    # save labels and best model weights to checkpoint folder
    with open(labels_dir, 'w') as outfile:
        json.dump(class_names, outfile)
    state = {
        'model': model_ft.state_dict(),
        'best_acc': best_acc,
    }
    torch.save(state, model_path)