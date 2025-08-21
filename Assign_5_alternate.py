import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from get_image_paths import get_image_paths


# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dataset constants
NUM_CLASSES = 21
DATA_PATH = '/users/student/pg/pg24/ritik/gnr/UCMerced_LandUse'
CATEGORIES = ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral',
              'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection',
              'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway',
              'sparseresidential', 'storagetanks', 'tenniscourt']

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
NUM_TRAIN_PER_CAT = 70
BATCH_SIZE = 16
IMAGE_SIZE = 128
EPOCHS = 200
LR = 1e-5
BETA = 0.001

class HVAE(nn.Module):
    def __init__(self):
        super(HVAE, self).__init__()
        # Encoder Layers
        #hidden_dims = [32, 64, 128, 256, 512]
        #channels = in_channels = 3
        #img_size = 128
        self.encoder_z1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  #[B, 3, 128, 128] -> [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2)
        )
        self.encoder_z2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 32, 64, 64] -> [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.encoder_z3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.encoder_z4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # -----------------------
        # Latent variables at each level:
        # -----------------------
        self.fc_mu1    = nn.Linear(32 * 64 * 64, 128)  # [B, 32*64*64] -> [B, 128]
        self.fc_logvar1= nn.Linear(32 * 64 * 64, 128)  # [B, 32*64*64] -> [B, 128]
        
        self.fc_mu2    = nn.Linear(64 * 32 * 32, 256)  # [B, 64*32*32] -> [B, 256]
        self.fc_logvar2= nn.Linear(64 * 32 * 32, 256)  # [B, 64*32*32] -> [B, 256]
        
        self.fc_mu3    = nn.Linear(128 * 16 * 16, 384) # [B, 128*16*16] -> [B, 384]
        self.fc_logvar3= nn.Linear(128 * 16 * 16, 384) # [B, 128*16*16] -> [B, 384]
        
        self.fc_mu4    = nn.Linear(256 * 8 * 8, 512)   # [B, 256*8*8] -> [B, 512]
        self.fc_logvar4= nn.Linear(256 * 8 * 8, 512)   # [B, 256*8*8] -> [B, 512]
        
        # -----------------------
        # Decoder: Hierarchical reconstruction
        # -----------------------
        self.fc_decoder4 = nn.Linear(512, 256 * 8 * 8)     # [B, 512] -> [B, 256*8*8]
        self.fc_decoder3 = nn.Linear(384, 128 * 16 * 16)   # [B, 384] -> [B, 128*16*16]
        self.fc_decoder2 = nn.Linear(256, 64 * 32 * 32)    # [B, 256] -> [B, 64*32*32]
        self.fc_decoder1 = nn.Linear(128, 32 * 64 * 64)    # [B, 128] -> [B, 32*64*64]
        
        self.conv_reduce1 = nn.Conv2d(256, 128, kernel_size=1)  
        self.conv_reduce2 = nn.Conv2d(128, 64, kernel_size=1) 
        self.conv_reduce3 = nn.Conv2d(64, 32, kernel_size=1) 
        
        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # [B, 32, 64, 64] -> [B, 3, 128, 128]
            nn.Tanh()
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        h1 = self.encoder_z1(x)  # [B, 3, 128, 128] -> [B, 32, 64, 64]
        h2 = self.encoder_z2(h1) # [B, 32, 64, 64] -> [B, 64, 32, 32]
        h3 = self.encoder_z3(h2) # [B, 64, 32, 32] -> [B, 128, 16, 16]
        h4 = self.encoder_z4(h3) # [B, 128, 16, 16] -> [B, 256, 8, 8]

        return [h1, h2, h3, h4]

    def decode(self, input, batch_size):
        z1, z2, z3, z4 = input

        d4 = self.fc_decoder4(z4)                                            # [B, 512] -> [B, 256*8*8]
        d4 = d4.view(batch_size, 256, 8, 8)                                  # [B, 256*8*8] -> [B, 256, 8, 8]
        d4 = nn.functional.interpolate(d4, scale_factor=2, mode='nearest')   # [B, 256, 8, 8] -> [B, 256, 16, 16]
        d4 = self.conv_reduce1(d4)                                           # [B, 256, 16, 16] -> [B, 128, 16, 16]

        d3 = self.fc_decoder3(z3)                                            # [B, 384] -> [B, 128*16*16]
        d3 = d3.view(batch_size, 128, 16, 16)                                # [B, 128*16*16] -> [B, 128, 16, 16]
        d3 = d3 + d4                                                         # [B, 128, 16, 16] + [B, 128, 16, 16] -> [B, 128, 16, 16]
        d3 = nn.functional.interpolate(d3, scale_factor=2, mode='nearest')   # [B, 128, 16, 16] -> [B, 128, 32, 32]
        d3 = self.conv_reduce2(d3)                                           # [B, 128, 32, 32] -> [B, 64, 32, 32]

        d2 = self.fc_decoder2(z2)                                            # [B, 256] -> [B, 64*32*32]
        d2 = d2.view(batch_size, 64, 32, 32)                                         # [B, 256] -> [B, 64, 32, 32]
        d2 = d2 + d3                                                         # [B, 64, 32, 32] + [B, 64, 32, 32] -> [B, 64, 32, 32]
        d2 = nn.functional.interpolate(d2, scale_factor=2, mode='nearest')   # [B, 64, 32, 32] -> [B, 64, 64, 64]
        d2 = self.conv_reduce3(d2)                                           # [B, 64, 64, 64] -> [B, 32, 64, 64]

        d1 = self.fc_decoder1(z1)                                            # [B, 128] -> [B, 32*64*64]
        d1 = d1.view(batch_size, 32, 64, 64)                                         # [B, 32*64*64] -> [B, 32, 64, 64]
        d1 = d1 + d2                                                         # [B, 32, 64, 64] + [B, 32, 64, 64] -> [B, 32, 64, 64]
        recon = self.final_decoder(d1)                                       # [B, 32, 64, 64] -> [B, 3, 128, 128]
        return recon

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encoder
        h1, h2, h3, h4 = self.encode(x)                             # [B, 3, 128, 128] -> [[B, 32, 64, 64], [B, 64, 32, 32], [B, 128, 16, 16], [B, 256, 8, 8]]

        # Level 1 latent
        #h1 = h1.view(h1.size(0), -1)
        h1 = h1.view(batch_size, -1)
        #h1 = torch.flatten(h1, start_dim=1)                         # [B, 32, 64, 64] -> [B, 32*64*64]
        h1_mu = self.fc_mu1(h1)                                     # [B, 32*64*64] -> [B, 128]
        h1_logvar = self.fc_logvar1(h1)                             # [B, 32*64*64] -> [B, 128]
        z1 = self.reparameterize(h1_mu, h1_logvar)                  # [B, 128] -> [B, 128]
        
        # Level 2 latent
        h2 = h2.view(batch_size, -1)
        #h2 = torch.flatten(h2, start_dim=1)                         # [B, 64, 32, 32] -> [B, 64*32*32]
        h2_mu = self.fc_mu2(h2)                                     # [B, 64*32*32] -> [B, 256]
        h2_logvar = self.fc_logvar2(h2)                             # [B, 64*32*32] -> [B, 256]
        z2 = self.reparameterize(h2_mu, h2_logvar)                  # [B, 256] -> [B, 256]
        
        # Level 3 latent
        h3 = h3.view(batch_size, -1)
        #h3 = torch.flatten(h3, start_dim=1)                         # [B, 128, 16, 16] -> [B, 128*16*16]
        h3_mu = self.fc_mu3(h3)                                     # [B, 128*16*16] -> [B, 384]
        h3_logvar = self.fc_logvar3(h3)                             # [B, 128*16*16] -> [B, 384]
        z3 = self.reparameterize(h3_mu, h3_logvar)                  # [B, 384] -> [B, 384]
        
        # Level 4 latent
        h4 = h4.view(batch_size, -1)
        #h4 = torch.flatten(h4, start_dim=1)                         # [B, 256, 8, 8] -> [B, 256*8*8]
        h4_mu = self.fc_mu4(h4)                                     # [B, 256*8*8] -> [B, 512]
        h4_logvar = self.fc_logvar4(h4)                             # [B, 256*8*8] -> [B, 512]
        z4 = self.reparameterize(h4_mu, h4_logvar)                  # [B, 512] -> [B, 512]
        
        result = [z1, z2, z3, z4]                                   # [B, 128], [B, 256], [B, 384], [B, 512]
        
        # Decoder
        recons = self.decode(result, batch_size)                                # [B, 128] -> [B, 3, 128, 128]

        return recons, [(h4_mu, h4_logvar), (h3_mu, h3_logvar), (h2_mu, h2_logvar), (h1_mu, h1_logvar)]

    # def loss_function(self, *args, **kwargs):
    #     recons = args[0]
    #     input = args[1]
    #     z4_mu = args[2]
    #     z4_logvar = args[3]
    #     z3_mu = args[4]
    #     z3_logvar = args[5]
    #     z2_mu = args[6]
    #     z2_logvar = args[7]
    #     z1_mu = args[8]
    #     z1_logvar = args[9]
    #     z4 = args[10]
    #     z3 = args[11]
    #     z2 = args[12]
    #     z1 = args[13]
        
    #     beta = BETA

    #     kld_weight = kwargs['M_N']  # Account for the minibatch samples
        
    #     # Reconstruction loss
    #     recons_loss = F.mse_loss(recons, input, reduction='sum')
        
    #     z1_kld = -0.5 * torch.sum(1 + z1_logvar - z1_mu.pow(2) - z1_logvar.exp())
    #     z2_kld = -0.5 * torch.sum(1 + z2_logvar - z2_mu.pow(2) - z2_logvar.exp())
    #     z3_kld = -0.5 * torch.sum(1 + z3_logvar - z3_mu.pow(2) - z3_logvar.exp())
    #     z4_kld = -0.5 * torch.sum(1 + z4_logvar - z4_mu.pow(2) - z4_logvar.exp())
        
    #     # Total KL divergence
    #     kld_loss = z1_kld + z2_kld + z3_kld + z4_kld
        
    #     # Overall loss
    #     loss = recons_loss + beta * kld_weight * kld_loss

    #     return {'loss': loss, 'Reconstruction Loss': recons_loss, 'KLD': kld_loss}

    def generate(self, x, **kwargs):
        return self.forward(x)[0]


def denormalize(tensor):
    # Undo normalization: x * std + mean
    mean = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean


# Load dataset
def load_data():
    train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = get_image_paths(DATA_PATH, CATEGORIES, NUM_TRAIN_PER_CAT)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset

# Training the model
def train_hvae():
    model = HVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    num_epochs = EPOCHS
    train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset = load_data()
    
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
            outputs, lts = model(images)
            # loss_dict = model.loss_function(*outputs, M_N=1)
            # loss = loss_dict["loss"]
            recons_loss = F.mse_loss(outputs, images, reduction='sum')
            kld_loss = sum(
                -0.5 * torch.sum(1 + mu - logvar.pow(2) - mu.exp()) for mu, logvar in lts
            )
            loss = recons_loss + BETA * kld_loss
            
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update weights
            optimizer.step()

            total_loss += loss.item()
            total_kl += kld_loss.item()
            total_recon += recons_loss.item()
            # total_kl += loss_dict["KLD"].item()
            # total_recon += loss_dict["Reconstruction Loss"].item()

        # Scheduler step
        scheduler.step()

        avg_train_loss = total_loss / len(train_dataset)
        avg_train_kl = total_kl / len(train_dataset)
        avg_train_recon = total_recon / len(train_dataset)
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
                #loss_dict = model.loss_function(*outputs, M_N=1.0)
                # val_loss += loss_dict["loss"].item()
                # val_kl += loss_dict["KLD"].item()
                # val_recon += loss_dict["Reconstruction Loss"].item()
                recons_loss = F.mse_loss(outputs[0], images, reduction='sum')
                kld_loss = sum(
                    -0.5 * torch.sum(1 + mu - logvar.pow(2) - mu.exp()) for mu, logvar in outputs[1]
                )
                loss = recons_loss + BETA * kld_loss
                val_loss += loss.item()
                val_kl += kld_loss.item()
                val_recon += recons_loss.item()


        avg_val_recon = val_recon / len(val_dataset)
        avg_val_kl = val_kl / len(val_dataset)
        history['val_recon'].append(avg_val_recon)
        history['val_kl'].append(avg_val_kl)

        avg_val_loss = val_loss / len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_hvae_model.pth")
            print("Model saved at epoch", epoch + 1)

        # Save epoch reconstruction
        if (epoch + 1) % 10 == 0:
            save_epoch_reconstruction(model, train_loader, epoch + 1)
        
    print("Training complete. Best validation loss:", best_val_loss)
    # Load the best model
    model.load_state_dict(torch.load("best_hvae_model.pth"))
    evaluate_hvae(model, test_loader)
    plot_training_curves(history)

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
        plt.imshow(images[2*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons[2*i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")

    plt.suptitle("Original (top) vs Reconstruction (bottom)")
    plt.savefig("reconstructions_final.png")
    plt.show()

# Global variable to store our fixed images and their labels
fixed_category_images = None
fixed_category_labels = None

def save_epoch_reconstruction(model, val_loader, epoch):
    global fixed_category_images, fixed_category_labels
    model.eval()
    
    # Categories we want to display
    target_categories = ['airplane', 'baseballdiamond', 'beach', 'intersection', 'runway', 'buildings']
    target_category_ids = [CATE2ID[cat] for cat in target_categories]
    
    # Only collect images during the first call (or if we don't have them yet)
    if fixed_category_images is None:
        # Store images from desired categories
        category_images = {cat_id: [] for cat_id in target_category_ids}
        
        # Collect images from each category
        for images, labels in val_loader:
            for i, label in enumerate(labels):
                if label.item() in target_category_ids and len(category_images[label.item()]) < 1:
                    category_images[label.item()].append(images[i])
            
            # Check if we have all the images we need
            if all(len(imgs) == 1 for imgs in category_images.values()):
                break
        
        # Get one image from each category
        selected_images = []
        selected_labels = []
        for cat_id in target_category_ids:
            if category_images[cat_id]:
                selected_images.append(category_images[cat_id][0])
                selected_labels.append(cat_id)
        
        # If we couldn't find all categories, fill with random images
        if len(selected_images) < len(target_categories):
            random_images, random_labels = next(iter(val_loader))
            needed = len(target_categories) - len(selected_images)
            random_indices = torch.randperm(len(random_images))[:needed]
            for idx in random_indices:
                selected_images.append(random_images[idx])
                selected_labels.append(random_labels[idx].item())
        
        # Save these for future epochs
        fixed_category_images = torch.stack(selected_images)
        fixed_category_labels = selected_labels
        
        print(f"Fixed images selected for categories: {[CATEGORIES[label] for label in fixed_category_labels]}")
    
    # Use our saved images
    selected_images = fixed_category_images.to(device)
    
    # Generate reconstructions
    with torch.no_grad():
        recons, *_ = model(selected_images)
    
    # Denormalize for visualization
    selected_images_display = denormalize(selected_images)
    recons_display = denormalize(recons)
    
    # Plot original vs reconstruction
    n = len(selected_images)
    plt.figure(figsize=(15, 5))
    
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(selected_images_display[i].permute(1, 2, 0).cpu().numpy())
        # Display the actual category name
        category_name = CATEGORIES[fixed_category_labels[i]]
        plt.title(category_name, fontsize=10)
        ax.axis("off")
        
        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(recons_display[i].permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    
    plt.suptitle(f"Epoch {epoch}: Original (top) vs Reconstruction (bottom)")
    plt.tight_layout()
    plt.savefig(f"reconstructions/epoch_{epoch:02d}.png")
    plt.close()
# def save_epoch_reconstruction(model, val_loader, epoch):
#     model.eval()
#     images, _ = next(iter(val_loader))
#     images = images.to(device)

#     with torch.no_grad():
#         recons, *_ = model(images)

#     images = denormalize(images)
#     recons = denormalize(recons)

#     n = 6
#     plt.figure(figsize=(12, 4))
#     for i in range(n):
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(images[2*i].permute(1, 2, 0).cpu().numpy())
#         ax.axis("off")

#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(recons[2*i].permute(1, 2, 0).cpu().numpy())
#         ax.axis("off")

#     plt.suptitle(f"Epoch {epoch}: Original (top) vs Reconstruction (bottom)")
#     plt.tight_layout()
#     plt.savefig(f"reconstructions/epoch_{epoch:02d}.png")
#     plt.close()

if __name__ == '__main__':
    train_hvae()
