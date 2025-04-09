import os 
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from PIL import Image
from torchvision.transforms import transforms
from siamese_network import SiameseNetwork



class FacePairsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((105,105)),
            transforms.ToTensor(),
        ])
        self.user_dirs = os.listdir(root_dir)
        self.image_paths = self._get_all_images()
        
    def _get_all_images(self):
        paths = []
        for user in self.user_dirs:
            user_path = os.path.join(self.root_dir, user)
            if os.path.isdir(user_path):
                for img_name in os.listdir(user_path):
                    img_path = os.path.join(user_path, img_name)
                    paths.append((user, img_path))
        return paths
        
    def __getitem__(self,idx):
        user1, img1_path = self.image_paths[idx]
        img1 = Image.open(img1_path)
            
        # 50% same, 50% different
        if random.random() < 0.5:
            # same user
            user_imgs = [p for u, p in self.image_paths if u == user1 and p != img1_path]
            img2_path = random.choice(user_imgs)
            label = 0
                
        else: 
            # different user
            other_users = [u for u in self.user_dirs if u != user1]
            user2 = random.choice(other_users)
            user_imgs = [os.path.join(self.root_dir, user2, f) for f in os.listdir(os.path.join(self.root_dir, user2))]
            img2_path = random.choice(user_imgs)
            label = 1

        img2 = Image.open(img2_path)
            
        return self.transform(img1), self.transform(img2), torch.tensor([label], dtype=torch.float32)
        
    def __len__(self):
        return len(self.image_paths)
            
# Contrastive Loss Function
# This loss function is used to train the Siamese network.
# It encourages the model to minimize the distance between similar images
# and maximize the distance between dissimilar images.
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, dist, label):
        loss = (1 - label) * dist.pow(2) + label * torch.clamp(self.margin - dist, min=0.0).pow(2)
        return loss.mean()

# Training the Siamese Network
# This function initializes the dataset and dataloader,
# sets up the model, loss function, and optimizer,
# and trains the model for a specified number of epochs.
def train():
    dataset = FacePairsDataset('data/users')
    loader = DataLoader(dataset, shuffle=True, batch_size=16)
    model = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 20
    for epoch in range(num_epochs):
        total_loss = 0
        for img1, img2, label in loader:
            output = model(img1, img2)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")


    if not os.path.exists('models/'):
        torch.save(model.state_dict(), "models/siamese_model.pt")
    else:
        os.mkdir('models/')
        torch.save(model.state_dict(), "models/siamese_model.pt")
    print("Model saved to models/siamese_model.pt")

if __name__ == "__main__":
    train()