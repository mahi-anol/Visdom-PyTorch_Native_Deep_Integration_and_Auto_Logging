import lightning as L
from src.Task1.cnn_model import SimpleCNN 
from lightning_cnn_model import LitCNN
from src.Task1.dataset import get_dataloaders # Use the function that returns both
from visdom_lightning_logger import SimpleVisdomLogger

def train():
    # Intializing dataloaders
    train_loader, val_loader = get_dataloaders()

    #  Initializing Model & Logger
    model = LitCNN(SimpleCNN())
    vis_logger = SimpleVisdomLogger(env_name="MNIST_Lightning_Run")

    # Trainer setup
    trainer = L.Trainer(
        max_epochs=3,
        logger=vis_logger,
        log_every_n_steps=5, # Logs every 5 batches
        deterministic=True
    )

    # start training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    train()