import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

#############################################
#        Preprocessed Dataset Class       #
#############################################

class PreprocessedDataset(Dataset):
    def __init__(self, base_dir, num_frames=16, transform=None):
        """
        Args:
            base_dir (str): Base directory for the split (e.g., "preprocessed/train").
            num_frames (int): Number of frames per video.
            transform: Optional transformation to apply.
        """
        self.samples = []  # Each sample is (frames_path, landmarks_path, label)
        self.num_frames = num_frames
        self.transform = transform
        
        # Expect subdirectories for each label.
        for label in os.listdir(base_dir):
            label_dir = os.path.join(base_dir, label)
            if not os.path.isdir(label_dir):
                continue
            frames_dir = os.path.join(label_dir, "frames")
            landmarks_dir = os.path.join(label_dir, "landmarks")
            if not os.path.isdir(frames_dir) or not os.path.isdir(landmarks_dir):
                continue
            for file in os.listdir(frames_dir):
                if file.endswith('.npy'):
                    frames_path = os.path.join(frames_dir, file)
                    landmarks_path = os.path.join(landmarks_dir, file)
                    self.samples.append((frames_path, landmarks_path, int(label)))
                    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frames_path, landmarks_path, label = self.samples[idx]
        frames_np = np.load(frames_path)      # Shape: (num_frames, H, W, 3)
        landmarks_np = np.load(landmarks_path)  # Shape: (num_frames, 68, 2)
        # Convert frames to tensor (C, H, W) and normalize.
        frames = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0 for frame in frames_np]
        video_tensor = torch.stack(frames)     # (num_frames, C, H, W)
        landmarks_tensor = torch.tensor(landmarks_np, dtype=torch.float32)
        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor, landmarks_tensor, label

#############################################
#          Model Components               #
#############################################

# Graph Convolutional Layer (GCN)
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: (batch, num_nodes, in_features)
        x = torch.matmul(adj, x)  # Aggregate neighbor features.
        x = self.linear(x)
        return x

# WaveNet Block (for temporal modeling)
class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size=2, dilation=1):
        super(WaveNetBlock, self).__init__()
        self.conv_filter = nn.Conv1d(channels, channels, kernel_size,
                                     padding=(kernel_size - 1) * dilation,
                                     dilation=dilation)
        self.conv_gate = nn.Conv1d(channels, channels, kernel_size,
                                   padding=(kernel_size - 1) * dilation,
                                   dilation=dilation)
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        # x: (batch, channels, T)
        filter_out = torch.tanh(self.conv_filter(x)[:, :, :x.size(2)])
        gate_out = torch.sigmoid(self.conv_gate(x)[:, :, :x.size(2)])
        out = filter_out * gate_out
        skip = self.conv_skip(out)[:, :, :x.size(2)]
        res = self.conv_res(out)[:, :, :x.size(2)]
        res = res + x
        return res, skip

class WaveNetTemporal(nn.Module):
    def __init__(self, channels, kernel_size=2, num_layers=3):
        super(WaveNetTemporal, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(WaveNetBlock(channels, kernel_size, dilation))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum += skip
        out = self.relu(skip_sum)
        return out

#############################################
#       Fatigue Detection Model           #
#############################################

class FatigueDetectionModel(nn.Module):
    def __init__(self, num_landmarks=68, spatial_feature_dim=128, num_classes=3, num_frames=16):
        super(FatigueDetectionModel, self).__init__()
        self.num_frames = num_frames
        
        # Use MobileNetV3 Small.
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.mobilenet_features = mobilenet.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # For MobileNetV3 Small, final feature dimension is typically 576.
        mobile_feat_dim = 576
        self.landmark_proj = nn.Linear(2, mobile_feat_dim)
        
        # GCN layers for spatial processing.
        self.gcn1 = GCNLayer(in_features=mobile_feat_dim, out_features=spatial_feature_dim)
        self.gcn2 = GCNLayer(in_features=spatial_feature_dim, out_features=spatial_feature_dim)
        
        # WaveNet-inspired temporal module.
        self.wavenet = WaveNetTemporal(channels=spatial_feature_dim, kernel_size=2, num_layers=3)
        
        # Dropout layer for regularization.
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head.
        self.fc = nn.Linear(spatial_feature_dim, num_classes)
        
        # Self-adaptive (learnable) adjacency matrix.
        self.adaptive_adj = nn.Parameter(torch.eye(num_landmarks))
    
    def forward(self, images, landmarks, adj=None):
        """
        Args:
            images: Tensor of shape (batch, T, C, H, W)
            landmarks: Tensor of shape (batch, T, num_landmarks, 2)
            adj: (Optional) external adjacency matrix. If provided, it is ignored.
        Returns:
            logits: Classification logits.
            fused_features: Fused features for visualization.
        """
        batch, T, C, H, W = images.shape
        images = images.view(batch * T, C, H, W)
        feat_maps = self.mobilenet_features(images)
        feat_maps = self.global_pool(feat_maps)
        visual_features = feat_maps.view(batch, T, -1)  # (batch, T, mobile_feat_dim)
        
        # Project landmarks.
        landmark_features = self.landmark_proj(landmarks)  # (batch, T, num_landmarks, mobile_feat_dim)
        visual_features_expanded = visual_features.unsqueeze(2).expand_as(landmark_features)
        fused_features = visual_features_expanded + landmark_features  # (batch, T, num_landmarks, mobile_feat_dim)
        
        spatial_features = []
        for t in range(T):
            x = fused_features[:, t, :, :]  # (batch, num_landmarks, mobile_feat_dim)
            x = F.relu(self.gcn1(x, self.adaptive_adj))
            x = F.relu(self.gcn2(x, self.adaptive_adj))
            x = torch.mean(x, dim=1)  # (batch, spatial_feature_dim)
            spatial_features.append(x)
        spatial_features = torch.stack(spatial_features, dim=1)  # (batch, T, spatial_feature_dim)
        temporal_input = spatial_features.transpose(1, 2)  # (batch, spatial_feature_dim, T)
        wavenet_out = self.wavenet(temporal_input)           # (batch, spatial_feature_dim, T)
        temporal_features = torch.mean(wavenet_out, dim=2)     # (batch, spatial_feature_dim)
        
        # Apply dropout for improved generalization.
        temporal_features = self.dropout(temporal_features)
        logits = self.fc(temporal_features)                  # (batch, num_classes)
        return logits, fused_features

#############################################
#           Training Script               #
#############################################

def train_model():
    # Set directories for preprocessed training data.
    TRAIN_DIR = "preprocessed/train"
    NUM_FRAMES = 16
    BATCH_SIZE = 8
    NUM_EPOCHS = 60  # Maximum number of epochs.
    
    train_dataset = PreprocessedDataset(TRAIN_DIR, num_frames=NUM_FRAMES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FatigueDetectionModel(num_landmarks=68, spatial_feature_dim=128, num_classes=3, num_frames=NUM_FRAMES)
    model.to(device)
    
    model_path = f"best_fatigue_model_{NUM_FRAMES}.pth"
    if os.path.exists(model_path):
        print("Loading previous best model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_loss = float('inf')
    epochs_without_improve = 0
    patience = 3  # Stop training if no improvement in training loss for 3 epochs.
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        
        for frames, landmarks, labels in train_loader:
            frames = frames.to(device)           # (batch, T, C, H, W)
            landmarks = landmarks.to(device)       # (batch, T, 68, 2)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(frames, landmarks, None)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * frames.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        # Update learning rate scheduler.
        scheduler.step(train_loss)
        
        # Check for improvement in loss.
        if train_loss < best_loss:
            best_loss = train_loss
            epochs_without_improve = 0
            torch.save(model.state_dict(), model_path)
            print("Best model saved.")
        else:
            epochs_without_improve += 1
            print(f"No improvement for {epochs_without_improve} epoch(s).")
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                break

if __name__ == '__main__':
    train_model()