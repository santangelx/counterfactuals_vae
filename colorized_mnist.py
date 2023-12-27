from matplotlib import pyplot as plt

import torch
import argparse

import numpy as np

from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from torchvision.transforms.functional import to_pil_image as to_pil
from torchvision import datasets, transforms
from PIL import Image

torch.multiprocessing.set_start_method('spawn') 

def load_data(categories=[5, 6], train=False):
    dataset = "mnist"
    root = "../data/"
    
    transform = []
    transform.append(transforms.Resize(32))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)
    if x.size(0) == 1 else x))
    transform = transforms.Compose(transform)

    if train == True:
        data = datasets.MNIST(root, transform=transform, train=True, download=True)
    elif train == False:
        data = datasets.MNIST(root, transform=transform, train=False, download=True)

    accepted = []
    for idx in range(len(data)):
        x, y = data[idx]
        if y in categories:
            accepted.append(idx)

    c = categories
    data.target_transform = lambda x: c.index(x) if x in c else None
    subdataset = Subset(data, accepted)

    loader = DataLoader(subdataset,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        batch_size=1)

    return loader


def colorize(loader, colors, prob=0.1, train=True):
    colored_images = []
    labels = []

    n_classes = colors.shape[0]

    compl_cnt = 0

    for x, y in loader:
        x = x.squeeze()
        labels.append(y)

        if train == True:

            if torch.rand(1).item() < prob:
                y = 1 - y
                # y = torch.LongTensor(1).random_(0, n_classes).item()
                compl_cnt += 1

        else:
            y = 1 - y
            if torch.rand(1).item() < prob:
                y = -y + 1
                # y = torch.LongTensor(1).random_(0, n_classes).item()
                compl_cnt += 1

        x[0, :, :] = x[0, :, :].mul(colors[y, 0]).clamp(0.0, 1.0)
        x[1, :, :] = x[1, :, :].mul(colors[y, 1]).clamp(0.0, 1.0)
        x[2, :, :] = x[2, :, :].mul(colors[y, 2]).clamp(0.0, 1.0)

        colored_images.append(x)

    colored_images = torch.stack(colored_images)
    labels = torch.cat(labels)

    loader = DataLoader(TensorDataset(colored_images, labels),
                        shuffle=True,
                        num_workers=8,
                        pin_memory=True,
                        batch_size=1)

    print("Proportion of complement colors {}".format(compl_cnt / colored_images.size(0)))

    return loader

class LabelIndexDataset(object):
    def __init__(self, dataset, label=0, opposite=False):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset

        if opposite:
            condition = (dataset.labels != label)
        else:
            condition = (dataset.labels == label)

        self.accepted = condition.nonzero().view(-1).tolist()

    def __getitem__(self, index):
        i = self.indices[self.accepted[index]]
        xy = self.dataset[self.accepted[index]]
        return i, xy

    def __len__(self):
        return len(self.accepted)


def BigLoader(dataset):
    return DataLoader(dataset, batch_size=512, num_workers=4, shuffle=False)



def classification_accuracy(network, loader):
    network.eval()

    p = []
    t = []
    for (x, y) in loader:
        p.append(network(x).detach())
        t.append(y.long().view(-1))

    p = torch.cat(p).squeeze()
    t = torch.cat(t).squeeze()
    acc = (p.argmax(1) == t).float().mean().item()

    return acc


class ConvNetwork(torch.nn.Module):
    def __init__(self, dr=0.2, n_out=10):
        super(ConvNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=5)
        self.drop1 = torch.nn.Dropout2d(dr)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=5)
        self.drop2 = torch.nn.Dropout2d(dr)

        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 100)
        self.drop = torch.nn.Dropout(dr)

        self.fc2 = torch.nn.Linear(100, n_out, bias=False)

    def features(self, x):
        x = self.drop1(self.pool(self.relu(self.conv1(x))))
        x = self.drop2(self.pool(self.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        return x


seed = 34
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

n_epochs = 11
log_interval = 5
l2_include = 1

# load data for requested classes only ##########################
classes = [5, 6]  # list(range(10))
n_classes = len(classes)
colors = torch.rand(n_classes, 3)

train = load_data(categories=classes, train=True)
test = load_data(categories=classes, train=False)

# colorize train and test data with complementary colors ########
prob = 0.01
loader_tr_colored_te = colorize(test, colors, prob, train=True)
loader_colored_tr = colorize(train, colors, prob, train=True)
loader_colored_te = colorize(test, colors, prob, train=False)
print("done colorizing")



print("samples with train colors")
plt.figure(figsize=(20, 5))
i = 0
for (x, y) in loader_colored_tr:
        plt.subplot(1, 10, i + 1)
        x = x.squeeze()
        img_to_preview = to_pil(x.cpu())
        plt.imshow(img_to_preview)
        plt.axis('off')
        plt.title(y.item())
        i += 1
        if i == 10:
            break
#plt.show()


print("samples with test colors")
plt.figure(figsize=(20, 5))
i = 0
for (x, y) in loader_colored_te:
        plt.subplot(1, 10, i + 1)
        x = x.squeeze()
        img_to_preview = to_pil(x.cpu())
        plt.imshow(img_to_preview)
        plt.axis('off')
        plt.title(y.item())
        i += 1
        if i == 10:
            break
#plt.show()

# train begins here ##############################################
network = ConvNetwork(n_out=n_classes)
#network.cuda()
optimizer = torch.optim.SGD(network.parameters(), lr=1e-2)
loss = torch.nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    network.train()
    for (x_i, y_i) in loader_colored_tr:
        x_i = x_i
        y_i = y_i

        optimizer.zero_grad()
        l1 = loss(network(x_i), y_i)
        l1.backward()
        optimizer.step()

print("Done training")
if (epoch % log_interval == 0):
    network.eval()

    acc_tr = classification_accuracy(network, loader_colored_tr)
    acc_te = classification_accuracy(network, loader_colored_te)

    print("{} | {} | {:.5f} | {:.5f} | {:.5f} ".format(seed,
                                                            epoch,
                                                            prob,
                                                            acc_tr,
                                                            acc_te))