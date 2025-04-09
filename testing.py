import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleDenoisingNet().to(device)
model.load_state_dict(torch.load("./multiscale_denoising.pth", map_location=device))
model.eval()

transform32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
transform64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
transform_gt = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
display_resize = transforms.Resize((256, 256))

test_folder = "testing_data"
test_images = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith('.jpg')]
result_dir = "test_result"
os.makedirs(result_dir, exist_ok=True)

num_test = len(test_images)
fig, axes = plt.subplots(4, num_test + 1, figsize=(4 * (num_test + 1), 16))

row_labels = ["Original", "Noisy 32x32x3", "Noisy 64x64x3", "Output"]
for row in range(4):
    ax = axes[row, 0]
    ax.text(0.2, 0.5, row_labels[row], fontsize=11, ha='left', va='center')
    ax.axis('off')

for col, img_path in enumerate(test_images):
    img = Image.open(img_path).convert("RGB")
    original_disp = transform_gt(img)
    
    img32 = transform32(img)
    img64 = transform64(img)
    noisy32 = add_noise(img32)
    noisy64 = add_noise(img64)
    
    disp_noisy32 = display_resize(noisy32)
    disp_noisy64 = display_resize(noisy64)
    
    input32 = noisy32.unsqueeze(0).to(device)
    input64 = noisy64.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input32, input64).squeeze(0).cpu()
    
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    Image.fromarray((original_disp.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
        os.path.join(result_dir, f"{base_name}_original.jpg"))
    Image.fromarray((disp_noisy32.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
        os.path.join(result_dir, f"{base_name}_noisy32.jpg"))
    Image.fromarray((disp_noisy64.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
        os.path.join(result_dir, f"{base_name}_noisy64.jpg"))
    Image.fromarray((output.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(
        os.path.join(result_dir, f"{base_name}_denoised.jpg"))
    
    axes[0, col + 1].imshow(original_disp.permute(1, 2, 0))
    axes[0, col + 1].set_title(base_name, fontsize=12)
    axes[0, col + 1].axis('off')
    
    axes[1, col + 1].imshow(disp_noisy32.permute(1, 2, 0))
    axes[1, col + 1].axis('off')
    
    axes[2, col + 1].imshow(disp_noisy64.permute(1, 2, 0))
    axes[2, col + 1].axis('off')
    
    axes[3, col + 1].imshow(output.permute(1, 2, 0))
    axes[3, col + 1].axis('off')

plt.tight_layout()
plt.show()