import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from get_image_paths import get_image_paths

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset constants
NUM_CLASSES = 21
DATA_PATH = '/home/guest1/Rick/Assignment_5/UCMerced_LandUse'
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
              'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
              'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway',
              'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
NUM_TRAIN_PER_CAT = 70
batch_size = 32

class HVAE(nn.Module):
    def __init__(self, latent_dims=[1024, 512]):
        super(HVAE, self).__init__()
        self.latent_dims = latent_dims
        
        # Encoder Layers
        #hidden_dims = [32, 64, 128, 256, 512]
        #channels = in_channels = 3
        #img_size = 256
        #z1 Layer ===> [B, 3, 256, 256] -> [128, 32, 32]
        self.encoder_z1_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), #[B, 3, 256, 256] -> [B, 32, 128, 128]
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #[B, 32, 128, 128] -> [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #[B, 64, 64, 64] -> [B, 128, 32, 32]
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU()
        )
        
        self.fc_mu1 = nn.Linear(128 * 32 * 32, latent_dims[0]) #[B, 128*32*32] -> [B, 1024]
        self.fc_logvar1 = nn.Linear(128 * 32 * 32, latent_dims[0]) #[B, 128*32*32] -> [B, 1024]

        self.embed_z1_code = nn.Linear(latent_dims[0], 256*256) #[B, 1024] -> [B, 65536]
        self.embed_data = nn.Conv2d(3, 3, kernel_size=1) #[B, 3, 256, 256] -> [B, 3, 256, 256]

        #[B, 4, 256, 256] -> [B, 128, 32, 32]
        self.encoder_z2_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1), #channels = in_channels + 1 [B, 4, 256, 256] -> [B, 32, 128, 128]
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  #[B, 32, 128, 128] -> [B, 64, 64, 64]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #[B, 64, 64, 64] -> [B, 128, 32, 32]
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU()
        )
        
        self.fc_mu2 = nn.Linear(128 * 32 * 32, latent_dims[1])      #[B, 128*32*32] -> [B, 512]
        self.fc_logvar2 = nn.Linear(128 * 32 * 32, latent_dims[1])  #[B, 128*32*32] -> [B, 512]
        
        # Decoder Layers
        #z1 Layer (Conditional Prior p(z2|z1))
        self.recons_mu2 = nn.Linear(latent_dims[0], latent_dims[1])     #[B, 1024] -> [B, 512]
        self.recons_logvar2 = nn.Linear(latent_dims[0], latent_dims[1]) #[B, 1024] -> [B, 512]

        #z2 Layer
        self.debed_z2_code = nn.Linear(latent_dims[1], 2*128*16*16)       #[B, 512] -> [B, 2*128*16*16] = [B, 65536]
        self.debed_z1_code = nn.Linear(latent_dims[0], 2*128*16*16)       #[B, 1024] -> [B, 2*128*16*16]

        #[B, 128, 32, 32] -> [B, 32, 128, 128]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),#, output_padding=1),   #[B, 128, 32, 32] -> [B, 64, 64, 64]
            # nn.BatchNorm2d(64),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),#, output_padding=1),    #[B, 64, 64, 64] -> [B, 32, 128, 128]
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        #[B, 32, 128, 128] -> [B, 3, 256, 256]
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),#, output_padding=1),    #[B, 32, 128, 128] -> [B, 32, 256, 256]
            # nn.BatchNorm2d(32),
            # nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),                                             #[B, 32, 256, 256] -> [B, 3, 256, 256]
            nn.Tanh()
        )

    def encode_z1(self, input): 
        result = self.encoder_z1_layers(input)          #[B, 3, 256, 256] -> [B, 128, 32, 32]
        result = torch.flatten(result, start_dim=1)     #[B, 128, 32, 32] -> [B, 128*32*32]

        z1_mu = self.fc_mu1(result)                     #[B, 128*32*32] -> [B, 1024]
        z1_logvar = self.fc_logvar1(result)             #[B, 128*32*32] -> [B, 1024]
        return [z1_mu, z1_logvar]                       #2*[B, 1024]
    
    def encode_z2(self, input, z1):                     #input: [B, 3, 256, 256], z1: [B, 1024]
        x = self.embed_data(input)                          #[B, 3, 256, 256] -> [B, 3, 256, 256]

        z1_embedded = self.embed_z1_code(z1)                #[B, 1024] -> [B, 256*256]
        z1_embedded = z1_embedded.view(-1, 1, 256, 256)     #[B, 256*256] -> [B, 1, 256, 256]

        result = torch.cat((x, z1_embedded), dim=1)         #[B, 4, 256, 256]

        result = self.encoder_z2_layers(result)             #[B, 4, 256, 256] -> [B, 128, 32, 32]
        result = torch.flatten(result, start_dim=1)         #[B, 128, 32, 32] -> [B, 128*32*32]

        z2_mu = self.fc_mu2(result)                         #[B, 128*32*32] -> [B, 512]
        z2_logvar = self.fc_logvar2(result)                 #[B, 128*32*32] -> [B, 512]
        return [z2_mu, z2_logvar]                           #2*[B, 512] 
    
    def encode(self, input):
        z1_mu, z1_logvar = self.encode_z1(input)            #[B, 3, 256, 256] -> 2*[B, 1024]
        z1 = self.reparameterize(z1_mu, z1_logvar)          #2*[B, 1024] -> [B, 1024]

        z2_mu, z2_logvar = self.encode_z2(input, z1)        #[B, 3, 256, 256], [B, 1024] -> 2*[B, 512]
        return [z1_mu, z1_logvar, z2_mu, z2_logvar, z1]     #[[B, 1024], [B, 1024], [B, 512], [B, 512], [B, 1024]]
    
    def decode(self, input):
        result = self.decoder(input)                        #[B, 128, 32, 32] -> [B, 32, 128, 128]
        result = self.final_layer(result)                   #B[32, 128, 128] -> [B, 3, 256, 256]
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, input):
        z1_mu, z1_logvar, z2_mu, z2_logvar, z1 = self.encode(input)     #[[B, 1024], [B, 1024], [B, 512], [B, 512], [B, 1024]]
        z2 = self.reparameterize(z2_mu, z2_logvar)                      #[B, 512]

        debedded_z1 = self.debed_z1_code(z1)                            #[B, 1024] -> [B, 2*128*16*16]
        debedded_z2 = self.debed_z2_code(z2)                            #[B, 512] -> [B, 2*128*16*16]
        result = torch.cat((debedded_z2, debedded_z1), dim=1)           #[B, 2*128*16*16] + [B, 2*128*16*16] -> [B, 2*2*128*16*16]
        result = result.view(-1, 128, 32, 32)                           #[B, 2*2*128*16*16] -> [B, 128, 32, 32] 
        recons = self.decode(result)                                    #[B, 128, 32, 32] -> [B, 3, 256, 256]       

        return [recons, input, z2_mu, z2_logvar, z1_mu, z1_logvar, z2, z1]   #[[B, 3, 256, 256], [B, 3, 256, 256], [B, 512], [B, 512], [B, 1024], [B, 1024], [B, 512], [B, 1024]]
        
    # Reconstruction Loss + KL Loss
    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        z2_mu = args[2]
        z2_logvar = args[3]
        z1_mu = args[4]
        z1_logvar = args[5]
        z2 = args[6]
        z1 = args[7]
        
        beta = 0.00025
        #VANILLA HVAE LOSS
        # Reconstruct (decode) z1 into z2
        # z2 ~ p(z2|z1) [This for the loss calculation]
        # p(z2|z1) parameters
        z2_p_mu = self.recons_mu2(z1)
        z2_p_logvar = self.recons_logvar2(z1)

        kld_weight = kwargs['M_N']  # Account for the minibatch samples
        
        # Reconstruction loss
        recons_loss = F.mse_loss(recons, input)
        
        # KL[q(z1|x) || p(z1)] where p(z1) is standard normal
        z1_kld = torch.mean(-0.5 * torch.sum(1 + z1_logvar - z1_mu ** 2 - z1_logvar.exp(), dim=1), dim=0)
        
        # KL[q(z2|x) || p(z2|z1)]
        z2_kld = 0.5 * torch.mean(torch.sum(
            z2_p_logvar - z2_logvar + 
            ((z2_mu - z2_p_mu)**2 + z2_logvar.exp()) / z2_p_logvar.exp() - 1, 
            dim=1), dim=0)
        
        # Total KL divergence
        kld_loss = z1_kld + z2_kld
        
        # Overall loss
        loss = recons_loss + beta * kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction Loss': recons_loss, 'KLD': kld_loss}
    
    def sample(self, batch_size, current_device, **kwargs):
        z1 = torch.randn(batch_size, self.latent_dims[0]).to(current_device)    #[B, 128]
        
        z2_mu = self.recons_mu2(z1)                                             #[B, 128] -> [B, 64]
        z2_logvar = self.recons_logvar2(z1)                                     #[B, 128] -> [B, 64]
        z2 = self.reparameterize(z2_mu, z2_logvar)                              #[B, 64]        

        debedded_z2 = self.debed_z2_code(z2)                                    #[B, 64] -> [B, 2*128*16*16]
        debedded_z1 = self.debed_z1_code(z1)                                    #[B, 128] -> [B, 2*128*16*16]

        result = torch.cat([debedded_z2, debedded_z1], dim=1)                   #[B, 2*128*16*16] + [B, 2*128*16*16] -> [B, 2*2*128*16*16]
        result = result.view(-1, 128, 32, 32)                                   #[B, 2*2*128*16*16] -> [B, 128, 32, 32]

        samples = self.decode(result)                                           #[B, 128, 32, 32] -> [B, 3, 256, 256]

        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]
# KL Divergence Loss
# def kl_divergence(mu, logvar):
#     return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim=0)


# Load dataset
def load_data():
    train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scales to [-1, 1]
    ])
    
    train_images = torch.stack([transform(Image.open(path).convert('RGB')) for path in train_image_paths])
    train_labels_tensors = torch.tensor([CATE2ID[x] for x in train_labels])
    
    test_images = torch.stack([transform(Image.open(path).convert('RGB')) for path in test_image_paths])
    test_labels_tensors = torch.tensor([CATE2ID[x] for x in test_labels])

    val_images = torch.stack([transform(Image.open(path).convert('RGB')) for path in val_image_paths])
    val_labels_tensors = torch.tensor([CATE2ID[x] for x in val_labels])
    
    train_dataset = TensorDataset(train_images, train_labels_tensors)
    test_dataset = TensorDataset(test_images, test_labels_tensors)
    val_dataset = TensorDataset(val_images, val_labels_tensors)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

# Training the model
def train_hvae():
    model = HVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 2000
    train_loader, test_loader, val_loader = load_data()
    
    best_val_loss = float('inf')
    history = {'train_recon': [], 'train_kl': [], 'val_recon': [], 'val_kl': []}
    os.makedirs("reconstructions", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_kl = 0
        total_recon = 0

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            
            #M_N = images.size(0) / len(train_loader.dataset)  # Minibatch size / total dataset size (Not required since taking mean in loss)
            # Forward pass
            outputs = model(images)
            loss_dict = model.loss_function(*outputs, M_N=1.0)
            loss = loss_dict["loss"]
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_kl += loss_dict["KLD"].item()
            total_recon += loss_dict["Reconstruction Loss"].item()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_kl = total_kl / len(train_loader)
        avg_train_recon = total_recon / len(train_loader)
        history['train_recon'].append(avg_train_recon)
        history['train_kl'].append(avg_train_kl)

        model.eval()
        val_loss = 0
        val_kl = 0
        val_recon = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss_dict = model.loss_function(*outputs, M_N=1.0)
                val_loss += loss_dict["loss"].item()
                val_kl += loss_dict["KLD"].item()
                val_recon += loss_dict["Reconstruction Loss"].item()

        avg_val_recon = val_recon / len(val_loader)
        avg_val_kl = val_kl / len(val_loader)
        history['val_recon'].append(avg_val_recon)
        history['val_kl'].append(avg_val_kl)

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hvae_model.pth")
            print("Model saved at epoch", epoch + 1)

        # Save epoch reconstruction
        if (epoch + 1) % 100 == 0:
            save_epoch_reconstruction(model, val_loader, epoch + 1)
        
    print("Training complete. Best validation loss:", best_val_loss)
    # Load the best model
    model.load_state_dict(torch.load("best_hvae_model.pth"))
    evaluate_hvae(model, test_loader)
    plot_training_curves(history)

def denormalize(tensor):
    # Undo normalization: x * std + mean
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean
    #tensor = (tensor + 1) / 2  # Scale from [-1, 1] to [0, 1]
    #return tensor 

def plot_training_curves(history):
    epochs = range(1, len(history['train_recon']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_recon'], label='Train Recon')
    plt.plot(epochs, history['val_recon'], label='Val Recon')
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_kl'], label='Train KL')
    plt.plot(epochs, history['val_kl'], label='Val KL')
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

# Evaluate the model - just visualize reconstructions
def evaluate_hvae(model, test_loader):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)

    with torch.no_grad():
        recons, *_ = model(images)
    
    # Denormalize before visualization
    images = denormalize(images)
    recons = denormalize(recons)
    # Plot original vs reconstruction
    n = 6
    plt.figure(figsize=(12, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[5*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[5*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

    plt.suptitle("Original (top) vs Reconstruction (bottom)")
    plt.savefig("reconstructions_final.png")
    plt.show()

def save_epoch_reconstruction(model, val_loader, epoch):
    model.eval()
    images, _ = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        recons, *_ = model(images)

    images = denormalize(images)
    recons = denormalize(recons)

    n = 6
    plt.figure(figsize=(12, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[5*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[5*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

    plt.suptitle(f"Epoch {epoch}: Original (top) vs Reconstruction (bottom)")
    plt.tight_layout()
    plt.savefig(f"reconstructions/epoch_{epoch:02d}.png")
    plt.close()

if __name__ == '__main__':
    train_hvae()
