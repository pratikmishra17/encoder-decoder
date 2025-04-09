import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def add_noise(img, num_squares=30, square_size_range=(1, 3)):
    img_noisy = img.clone()
    _, H, W = img_noisy.shape
    for _ in range(num_squares):
        square_size = random.randint(square_size_range[0], square_size_range[1])
        top = random.randint(0, H - square_size)
        left = random.randint(0, W - square_size)
        noise_val = random.choice([0.0, 1.0])
        img_noisy[:, top:top+square_size, left:left+square_size] = noise_val
    return img_noisy

class ImageDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        self.gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.transform32 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.transform64 = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        gt = self.gt_transform(img)
        img32 = self.transform32(img)
        img64 = self.transform64(img)
        noisy32 = add_noise(img32, num_squares=30, square_size_range=(1, 3))
        noisy64 = add_noise(img64, num_squares=30, square_size_range=(1, 3))
        return noisy32, noisy64, gt

class MultiScaleDenoisingNet(nn.Module):
    def __init__(self):
        super(MultiScaleDenoisingNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Linear(256 * 8 * 8, 2048)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                           
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                
        )
        self.fc2 = nn.Linear(256 * 8 * 8, 2048)
    
        self.decoder_fc = nn.Linear(4096, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),     
            nn.Sigmoid()
        )
    def forward(self, x32, x64):
        batch_size = x32.size(0)
        out1 = self.encoder1(x32)                     
        feat1 = self.fc1(out1.view(batch_size, -1))   
        out2 = self.encoder2(x64)                     
        feat2 = self.fc2(out2.view(batch_size, -1))   
        features = torch.cat((feat1, feat2), dim=1)   
        dec_input = self.decoder_fc(features)         
        dec_input = dec_input.view(batch_size, 256, 8, 8)
        output = self.decoder(dec_input)              
        return output

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50
    batch_size = 4
    lr = 0.001  
    dataset = ImageDataset("training_data")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = MultiScaleDenoisingNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for noisy32, noisy64, gt in dataloader:
            noisy32 = noisy32.to(device)
            noisy64 = noisy64.to(device)
            gt = gt.to(device)
            
            optimizer.zero_grad()
            outputs = model(noisy32, noisy64)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * noisy32.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    torch.save(model.state_dict(), "./multiscale_denoising.pth")
    print("Training complete. Model saved as ./multiscale_denoising.pth")

if __name__ == "__main__":
    train()





# reference - https://discuss.pytorch.org/t/correct-way-to-define-two-encoder-modules/41432/2
# reference - https://medium.com/@ahmadsabry678/a-perfect-guide-to-understand-encoder-decoders-in-depth-with-visuals-30805c23659b