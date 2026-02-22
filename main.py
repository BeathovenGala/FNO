import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse

from simulator.generators import LensingDataset
from models.u_net import SimpleUNet

def train(epochs=10, batch_size=16, lr=1e-3, device='cuda'):
    print(f"--- Starting MVC Training on {device} ---")
    
    # 1. Data
    dataset = LensingDataset(epoch_size=1000, pixel_count=64)
    # num_workers=0 to avoid multiprocessing issues with our physics engine for now
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
    
    # 2. Model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 3. Loop
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, true_kappas) in enumerate(loader):
            images = images.to(device)
            true_kappas = true_kappas.to(device)
            
            # Forward
            pred_kappas = model(images)
            
            # Loss (Supervised for MVC)
            loss = criterion(pred_kappas, true_kappas)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Visualize last batch
        if (epoch + 1) % 5 == 0:
            visualize(images, true_kappas, pred_kappas, epoch+1)

    print("Training Complete.")
    
    # Save Model
    torch.save(model.state_dict(), "mvc_unet.pth")
    print("Model saved to mvc_unet.pth")

def visualize(img, true_k, pred_k, epoch):
    """Saves a comparison plot."""
    # Detach and CPU
    img = img[0, 0].detach().cpu()
    tk = true_k[0, 0].detach().cpu()
    pk = pred_k[0, 0].detach().cpu()
    
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='inferno')
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(tk, cmap='viridis')
    plt.title("True Kappa")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(pk, cmap='viridis')
    plt.title(f"Pred Kappa (E{epoch})")
    plt.axis('off')
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/epoch_{epoch}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, device=args.device)
