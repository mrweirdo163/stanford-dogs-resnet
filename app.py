import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from data.load import load_datasets
from torch.utils.data import DataLoader
import configs
import os

import matplotlib.pyplot as plt

def main():
    batch_size = 64
    epochs = 10
    plot_path = configs.plots
    #Load the datasets
    train_data, test_data, classes = load_datasets("stanford_dogs")

    #Create data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = np.swapaxes(np.swapaxes(images.numpy(), 1, 2), 2, 3)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(classes[labels[idx]], {'fontsize': batch_size/5}, pad=0.4)
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Initial_Visualization"))
    else:
        plt.show()
    plt.clf()

    #Load the Resnet model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes=len(classes))

    #Move model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Transfer Learning setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/10 completed.")

    #Evaluation
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            accuracy += torch.sum(preds == labels).item()
    accuracy = accuracy / len(test_data) * 100
    print(f"Test accuracy: {accuracy * 100:.2f}%")

main()