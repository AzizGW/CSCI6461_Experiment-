#!/usr/bin/env python
# coding: utf-8

# In[48]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



# In[49]:


# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# # Train the model using CPU only

# In[50]:


import psutil
# To train PyTorch models on GPUs on Apple Silicon, set Metal Performance Shaders (MPS) as the backend.
device = torch.device("cpu")

# Instantiate the model and move it to the GPU
model = Net().to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
import time
start_time = time.time()
step_time_sum = 0
epoch_time_sum = 0
trainloaderSum = 0
cpu_percentSum = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        trainloaderSum += len(trainloader)
        step_start_time = time.time()

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 50 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset), 100. * batch_idx / len(trainloader), running_loss / (batch_idx + 1), 100. * correct / total))
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_time_sum += step_time
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_time_sum += epoch_time
    cpu_percentSum += psutil.cpu_percent()

   




# Record the end time
end_time = time.time()




# In[51]:


avg_epoch_time = epoch_time_sum / num_epochs
print(f"Avg Seconds / Epoch: {avg_epoch_time}")


avg_step_time = step_time_sum / trainloaderSum

avg_step_time = format(avg_step_time, ".10f")
print(f"Avg Seconds / Step: {avg_step_time}")

cpu_percent = format(cpu_percentSum/num_epochs, ".10f")
print(f"CPU Utilization: {cpu_percent}%")

# Calculate the total training time
training_time = end_time - start_time

print(f"Training finished (CPU only). Total training time: {training_time:.2f} seconds.")





# In[23]:


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the model on the 10000 test images: %.2f%%' % accuracy)


# In[24]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the MNIST test dataset and set the device to CPU (as before)

# Evaluate the model and make predictions
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# # Now let's train the model using GPU

# In[46]:


import psutil
# To train PyTorch models on GPUs on Apple Silicon, set Metal Performance Shaders (MPS) as the backend.
device = torch.device("mps")

# Instantiate the model and move it to the GPU
model = Net().to(device)

# Set the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
import time
start_time = time.time()
step_time_sum = 0
epoch_time_sum = 0
trainloaderSum = 0
cpu_percentSum = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(trainloader, 0):
        trainloaderSum += len(trainloader)
        step_start_time = time.time()

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 50 == 0:
            print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                epoch, batch_idx * len(inputs), len(trainloader.dataset), 100. * batch_idx / len(trainloader), running_loss / (batch_idx + 1), 100. * correct / total))
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        step_time_sum += step_time
    
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_time_sum += epoch_time
    cpu_percentSum += psutil.cpu_percent()

   




# Record the end time
end_time = time.time()


# In[47]:


avg_epoch_time = epoch_time_sum / num_epochs
print(f"Avg Seconds / Epoch: {avg_epoch_time}")


avg_step_time = step_time_sum / trainloaderSum

avg_step_time = format(avg_step_time, ".10f")
print(f"Avg Seconds / Step: {avg_step_time}")

cpu_percent = format(cpu_percentSum/num_epochs, ".10f")
print(f"CPU Utilization: {cpu_percent}%")

# Calculate the total training time
training_time = end_time - start_time

print(f"Training finished (heterogeneous). Total training time: {training_time:.2f} seconds.")





# In[60]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the MNIST test dataset and set the device to CPU (as before)

# Evaluate the model and make predictions
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Calculate metrics
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# In[62]:


# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print('Accuracy of the model on the 10000 test images: %.2f%%' % accuracy)


# In[59]:


import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['Total Training Time', 'Avg Seconds per Epoch', 'Avg (microseconds) per Step ', 'CPU Utilization', 'Test Accuracy']
cpu_only = [285.44, 14.2717, 12.6435, 28.60, 98.95]
heterogeneous = [229.27, 11.4636, 9.5036, 14.24, 98.94]
percentage_difference = [20.05, 19.67, 24.84, 50.2, '']

# Bar chart
bar_width = 0.3
index = np.arange(len(metrics))

fig, ax = plt.subplots()

bar1 = ax.bar(index - bar_width / 2, cpu_only, bar_width, label='CPU Only')
bar2 = ax.bar(index + bar_width / 2, heterogeneous, bar_width, label='Heterogeneous')

# Add percentage difference as text above the bars
for i in range(len(metrics)):
    if percentage_difference[i] != '':
        ax.text(i + bar_width / 2, max(cpu_only[i], heterogeneous[i]) * 1.01, f'{percentage_difference[i]}%', ha='center', va='bottom')

# Customize chart
ax.set_ylabel('Value')
ax.set_title('Comparison of \'CPU Only\' and \'Heterogeneous\' Results')
ax.set_xticks(index)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend()

fig.tight_layout()
plt.show()


# In[ ]:




