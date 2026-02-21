import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from src.Task1.cnn_model import SimpleCNN
from src.Task1.dataset import get_dataloaders  

# Constants & Initialization 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

def run_phase(model, loader, criterion, optimizer=None):
    """Universal function for a training or validation pass"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0

    # Gradient context depends on whether we are training or validating
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            if is_train: 
                optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, target)
            
            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return running_loss / len(loader), 100. * correct / total

def train():
    # Set Visdom client
    vis = visdom.Visdom()
    try:
        if not vis.check_connection():
            raise
    except:
        print("Error: Visdom server not running. Run 'python -m visdom.server' in a terminal.")
        return 

    # Prepare Data & Model 
    train_loader, val_loader = get_dataloaders()
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training started on {DEVICE} for {EPOCHS} epochs...")

    # Training Loop 
    for epoch in range(1, EPOCHS + 1):
        # Run Training and Validation phases
        t_loss, t_acc = run_phase(model, train_loader, criterion, optimizer)
        v_loss, v_acc = run_phase(model, val_loader, criterion)

        # Manual Visdom Logging 
        # Log Losses (Train vs Val)
        vis.line(
            X=torch.tensor([epoch]), Y=torch.tensor([t_loss]), 
            win='loss', name='train', update='append' if epoch > 1 else None,
            opts=dict(title='Loss', showlegend=True, xlabel='Epoch', ylabel='Loss')
        )
        vis.line(
            X=torch.tensor([epoch]), Y=torch.tensor([v_loss]), 
            win='loss', name='val', update='append'
        )

        # Log Accuracies (Train vs Val)
        vis.line(
            X=torch.tensor([epoch]), Y=torch.tensor([t_acc]), 
            win='acc', name='train', update='append' if epoch > 1 else None,
            opts=dict(title='Accuracy (%)', showlegend=True, xlabel='Epoch', ylabel='Accuracy')
        )
        vis.line(
            X=torch.tensor([epoch]), Y=torch.tensor([v_acc]), 
            win='acc', name='val', update='append'
        )

        print(f"Epoch {epoch} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Train Acc | {t_acc:.4f} | Val Acc: {v_acc:.2f}%")

if __name__ == '__main__':
    train()