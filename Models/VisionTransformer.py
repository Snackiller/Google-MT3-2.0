import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from transformers import ViTModel, ViTConfig, ViTImageProcessor
from time import time
from mtutil import SpectrogramNoteEventDataset
from mtconfig import SEED, SPECTROGRAM_DIR, CSV_DIR
from mtmodels import ViTForRegression

# Set random seed for reproducibility
torch.manual_seed(SEED)

# Load model configuration
config = ViTConfig.from_pretrained('google/vit-base-patch16-224')

<<<<<<< Updated upstream
class ViTForRegression(nn.Module):
    def __init__(self, vit_model, config, num_features):
        super(ViTForRegression, self).__init__()
        self.vit = vit_model
        self.regressor = nn.Linear(config.hidden_size, num_features)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values)
        sequence_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        logits = self.regressor(sequence_output)
        return logits

=======
>>>>>>> Stashed changes
# Initialize the Vision Transformer model
model = ViTModel(config)
model = ViTForRegression(model, config, num_features=50)
print(model)
model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Initialize the image processor
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

print("Loading dataset...")
dataset = SpectrogramNoteEventDataset(SPECTROGRAM_DIR, CSV_DIR, n_events=10, transform=transform)
print(f"Dataset loaded with {len(dataset)} samples")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training and Validation Functions
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets, _, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 50)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 50)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    return running_loss / len(dataloader)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 10
start_time = time()
for epoch in range(num_epochs):
    epoch_start_time = time()
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Time taken in thie epoch: {time() - epoch_start_time:.2f}s")
print(f"Total time taken: {time() - start_time:.2f}s")

# Model Saving
model_path = "vit_music_transcription.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
