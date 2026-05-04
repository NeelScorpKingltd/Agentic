import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np  

# =============================================
# Device Configuration
# =============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =============================================
# Hyperparameters & configurations
# =============================================
# Learning rate
learning_rate = 0.001

# Number of epochs
num_epochs = 5

# Batch size for training
batch_size = 256

# Dropout probability (for regularization)
dropout_prob = 0.2

# =============================================
# Data Preprocessing
# ============================================= 
# Transformations for the training and testing data     
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1] for better convergence
])

# Load the MNIST dataset for training and testing
train_dataset = torchvision.datasets.MNIST(root='ANN/data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='ANN/data', train=False, transform=transform, download=True)
# Data loaders for batching and shuffling the data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#============================================
# Model Definition
#===========================================
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        # Images are 28x28 pixels. Flattening them gives 28 * 28 = 784 input features.
        self.flatten = nn.Flatten()

        #defining the layers sequentially
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),  # First hidden layer with 128 neurons
            nn.ReLU(),                # Activation function
            nn.Dropout(dropout_prob), # Dropout for regularization
            nn.Linear(128, 10)          # Output layer with 10 neurons (for 10 classes)
        )
    def forward(self, x):
        #this defines how the data flows through the network
        x = self.flatten(x)  # Flatten the input images
        out = self.layers(x)  # Pass the data through the defined layers
        return out
    
# =============================================
# Model Initialization
# ============================================= 
model = ANN().to(device)
print(model)
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer adapts the learning rate during training and generally

# =============================================
# Training Loop
# =============================================
train_losses = []  # To store the average loss for each epoch
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to the appropriate device

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients from the previous step
        loss.backward()        # Compute gradients
        optimizer.step()       # Update model parameters

        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# =============================================
# Plotting the training loss
# ============================================= 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# =============================================
# Evaluation on the test set    
# =============================================
print("\n Evaluating the model on the test set...")
model.eval()  # Set the model to evaluation mode    
with torch.no_grad():  # No need to compute gradients during evaluation
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# =============================================
#visualize incorrect predictions using matplotlib
incorrect_images = []
incorrect_labels = []           
model.eval()  # Set the model to evaluation mode    
with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
        for i in range(len(labels)):
            if predicted[i] != labels[i]:  # If the prediction is incorrect
                incorrect_images.append(images[i].cpu())  # Store the image (move to CPU for visualization)
                incorrect_labels.append((predicted[i].item(), labels[i].item()))  # Store predicted and true labels 
# Visualize some incorrect predictions
num_incorrect_to_show = 10  
plt.figure(figsize=(15, 5))
for i in range(num_incorrect_to_show):  
    if i >= len(incorrect_images):
        break  # If there are fewer incorrect images than the number we want to show
    plt.subplot(2, 5, i + 1)
    plt.imshow(incorrect_images[i].squeeze(), cmap='gray')  # Show the image (remove channel dimension)
    plt.title(f'Pred: {incorrect_labels[i][0]}, True: {incorrect_labels[i][1]}')  # Show predicted and true labels
    plt.axis('off')
plt.tight_layout()
plt.show()



