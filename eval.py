import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image
import os

# Load the saved model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(1024 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 1024 * 2 * 2)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Net()
model.load_state_dict(torch.load('my_model.pth'))
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # ensure the model is on the right device
model.eval()

# Define the transform for test images
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :] if x.shape[0]==4 else x),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the test dataset
test_dataset = ImageFolder('eval', transform=data_transform)

# Create a list to store the predictions and labels
predictions = []
class_labels = ['DOG', 'CAT']  # get the class labels

# Iterate over the test images and make predictions
true_classes = []
for image_path, true_class in test_dataset.samples:
    image = Image.open(image_path)
    image_tensor = data_transform(image).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # ensure the data is on the right device
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)
    predictions.append(class_labels[predicted_class.item()])
    true_classes.append(class_labels[true_class])

# Print the predictions with labels
for image_path, prediction in zip([x[0] for x in test_dataset.samples], predictions):
    class_name = os.path.basename(os.path.dirname(image_path))
    print(f"Class: {class_name} - Predicted Class: {prediction}")

correct_predictions = sum([true == pred for true, pred in zip(true_classes, predictions)])
accuracy = correct_predictions / len(true_classes)
print(f'Accuracy: {accuracy * 100}%')