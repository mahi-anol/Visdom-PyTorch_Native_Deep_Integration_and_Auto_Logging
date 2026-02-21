import torch
import torch.nn as nn
import torch.optim as optim
from src.Task1.cnn_model import SimpleCNN
from src.Task1.dataset import get_dataloaders  # Updated import
from src.Task5.visdom_gradient_logger import VisdomGradientLogger

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return val_loss / len(loader), 100. * correct / len(loader.dataset)

def train_with_auto_logging():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Initializing dataloaders/
    train_loader, val_loader = get_dataloaders()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialization of the Deep Integration Logger
    grad_logger = VisdomGradientLogger(model)

    print(f"Training on {device}... Check Visdom at http://localhost:8097")
    
    for epoch in range(1, 3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # The backward pass will triger hooks automatically
            loss.backward()
            optimizer.step()
            
            # Process and send the gradient data to Visdom
            # This is the "Refined" method that avoids network spam
            grad_logger.log_step() 

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item():.6f}")

        # Validation Phase at the end of Epoch 
        v_loss, v_acc = validate(model, val_loader, criterion, device)
        print(f"\n--> Epoch {epoch} Validation: Average Loss: {v_loss:.4f}, Accuracy: {v_acc:.2f}%\n")
        
        # grad_logger.vis.line() 

    # Cleanup
    grad_logger.cleanup()

if __name__ == "__main__":
    train_with_auto_logging()