import torch
from torchvision import transforms
from PIL import Image
import os
import gc  # For garbage collection

# Define your image processing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256 (adjust if needed)
    transforms.ToTensor()           # Convert the image to a PyTorch tensor (automatically normalizes between 0-1)
])

# Function to load and process a single image from disk
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
    image = transform(image)  # Apply defined transformations
    return image

# Dummy model (replace with your actual model)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3)  # Example conv layer
        self.fc1 = torch.nn.Linear(16 * 254 * 254, 10)  # Example fully connected layer (adjust as needed)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU activation after conv
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc1(x)  # Fully connected output layer
        return x

# Main function to create batches, train the model, and manage memory
def process_dataset(image_dir, model, criterion, optimizer, batch_size):
    model.train()  # Set the model to training mode
    
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]  # Adjust image format if needed
    
    batch = []  # Initialize the batch list
    total_loss = 0  # To keep track of the total loss

    for idx, image_path in enumerate(image_paths):
        # Load an image and add a batch dimension
        image_tensor = load_image(image_path).unsqueeze(0)  # unsqueeze to add batch dimension
        batch.append(image_tensor)

        # Once the batch is full, process it
        if len(batch) == batch_size:
            batch_tensor = torch.cat(batch).to(device)  # Stack the images into a batch and move to GPU

            # Generate dummy labels (replace with actual labels in your use case)
            labels = torch.randint(0, 10, (batch_size,)).to(device)  # Random dummy labels

            # Forward pass through the model
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(batch_tensor)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            # Backward pass and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Track the total loss
            total_loss += loss.item()

            # Clear memory
            del batch_tensor, outputs, loss  # Delete tensors to free memory
            torch.cuda.empty_cache()  # Clear unused memory from the GPU
            gc.collect()  # Trigger garbage collection to free memory
            batch = []  # Reset the batch list for the next set of images

    # Handle any remaining images in the last partial batch
    if len(batch) > 0:
        batch_tensor = torch.cat(batch).to(device)  # Stack remaining images into a batch
        labels = torch.randint(0, 10, (len(batch),)).to(device)  # Adjust labels size for remaining batch

        optimizer.zero_grad()  # Clear gradients
        outputs = model(batch_tensor)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model

        total_loss += loss.item()  # Track loss for the last batch

        # Clear memory for the final batch
        del batch_tensor, outputs, loss  # Delete tensors
        torch.cuda.empty_cache()  # Clear GPU cache
        gc.collect()  # Trigger garbage collection

    return total_loss / len(image_paths)  # Return average loss

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Initialize the model, loss function, and optimizer
model = SimpleModel().to(device)  # Move the model to the GPU
criterion = torch.nn.CrossEntropyLoss()  # Loss function (adjust as per your task)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for the model

# Dataset and training setup
image_directory = "/path/to/your/image_folder"  # Path to your dataset
batch_size = 32  # Set the batch size (adjust as per memory limits)
num_epochs = 5  # Number of epochs

# Main training loop
for epoch in range(num_epochs):
    avg_loss = process_dataset(image_directory, model, criterion, optimizer, batch_size)  # Process dataset for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  # Print average loss for the epoch
