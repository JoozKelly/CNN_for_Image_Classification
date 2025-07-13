import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms


class NeuralNet(nn.Module):
    def __init__(self):
        super(). __init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def classify_image(img):
    net= NeuralNet()
    net.load_state_dict(torch.load('trained_net.pth'))

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    
    net.eval()
    with torch.no_grad():
        output = net(img)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]