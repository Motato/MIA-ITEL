import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==========================================
# Arquitectura del VAE (Basado en materiales del curso)
# ==========================================
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, 2)
        self.fc_logvar = nn.Linear(256, 2)
        # Decoder
        self.fc2 = nn.Linear(2, 256)
        self.fc3 = nn.Linear(256, 784)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Función de pérdida VAE (Reconstrucción + Divergencia KL)
def vae_loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ==========================================
# Pipeline Principal
# ==========================================
def run_generative_pipeline():
    os.makedirs('resultados', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Cargar Dataset
    transform = transforms.ToTensor()
    dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 2. Inicializar Modelo
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 20
    loss_history = []

    # 3. Bucle de Entrenamiento
    print(f"Entrenando VAE por {epochs} épocas en {device}...")
    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.view(-1, 784).to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        avg_loss = train_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        print(f"Época {epoch}/{epochs} | Pérdida Promedio: {avg_loss:.2f}")

    # 4. Graficar y guardar pérdida
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', color='red')
    plt.title('Pérdida de Entrenamiento VAE (Loss Function)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.grid(True)
    plt.savefig('resultados/vae_loss_graph.png')
    plt.close()

    # 5. Comparación Real vs Reconstruido
    model.eval()
    data, _ = next(iter(dataloader))
    data = data.view(-1, 784).to(device)
    with torch.no_grad():
        recon, _, _ = model(data)
    
    plt.figure(figsize=(10, 4))
    for i in range(5): # Mostrar 5 reales y 5 reconstruidas
        # Reales
        plt.subplot(2, 5, i+1)
        plt.imshow(data[i].cpu().view(28,28), cmap="gray")
        plt.title("Real")
        plt.axis("off")
        # Generadas (Reconstruidas)
        plt.subplot(2, 5, i+6)
        plt.imshow(recon[i].cpu().view(28,28), cmap="gray")
        plt.title("Generado")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('resultados/vae_real_vs_generado.png')
    plt.close()

    # 6. Generar 10 nuevas imágenes sintéticas desde ruido aleatorio
    print("Generando 10 muestras completamente sintéticas...")
    with torch.no_grad():
        # Muestreamos 10 puntos aleatorios del espacio latente (2 dimensiones)
        z_random = torch.randn(10, 2).to(device) 
        synthetic_images = model.decode(z_random).cpu()

    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(synthetic_images[i].view(28,28), cmap="gray")
        plt.title(f"Sintética {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('resultados/vae_10_sinteticas.png')
    plt.show()

if __name__ == "__main__":
    run_generative_pipeline()
    
