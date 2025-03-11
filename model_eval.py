import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import collections

#############################################
#         Preprocessed Dataset Class      #
#############################################

class PreprocessedDataset(Dataset):
    def __init__(self, base_dir, num_frames=16, transform=None):
        """
        Args:
            base_dir (str): Base directory for the split (e.g., "preprocessed/test").
            num_frames (int): Number of frames per video (should match the preprocessed value).
            transform: Optional transformation to apply.
        """
        self.samples = []  # Each sample is (frames_path, landmarks_path, label)
        self.num_frames = num_frames
        self.transform = transform
        
        # Assume subdirectories for each label.
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
        frames_np = np.load(frames_path)      # (num_frames, H, W, 3)
        landmarks_np = np.load(landmarks_path)  # (num_frames, 68, 2)
        # Convert frames to tensor with shape (num_frames, C, H, W) and normalize.
        frames = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0 
                  for frame in frames_np]
        video_tensor = torch.stack(frames)
        landmarks_tensor = torch.tensor(landmarks_np, dtype=torch.float32)
        if self.transform:
            video_tensor = self.transform(video_tensor)
        return video_tensor, landmarks_tensor, label

#############################################
#           Model Components              #
#############################################

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x, adj):
        # x: (batch, num_nodes, in_features)
        x = torch.matmul(adj, x)  # Aggregate neighbor features.
        x = self.linear(x)
        return x

class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                padding=(kernel_size - 1) * dilation,
                                dilation=dilation)
    def forward(self, x):
        out = self.conv1d(x)
        out = out[:, :, :x.size(2)]
        return out

# (Optional: Here we use a WaveNet-inspired temporal module in training; 
# you may choose to keep the same structure in evaluation.)
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
            skip_sum = skip_sum + skip
        out = self.relu(skip_sum)
        return out

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
        
        # Classification head.
        self.fc = nn.Linear(spatial_feature_dim, num_classes)
        
        # Self-adaptive (learnable) adjacency matrix.
        self.adaptive_adj = nn.Parameter(torch.eye(num_landmarks))
    
    def forward(self, images, landmarks, adj=None):
        """
        Args:
            images: Tensor of shape (batch, T, C, H, W)
            landmarks: Tensor of shape (batch, T, num_landmarks, 2)
            adj: Optional external adjacency matrix. If None, uses self.adaptive_adj.
        Returns:
            logits: Classification logits.
            fused_features: Fused features of shape (batch, T, num_landmarks, mobile_feat_dim)
                            used for visualization.
        """
        batch, T, C, H, W = images.shape
        images = images.view(batch * T, C, H, W)
        feat_maps = self.mobilenet_features(images)
        feat_maps = self.global_pool(feat_maps)
        visual_features = feat_maps.view(batch, T, -1)  # (batch, T, mobile_feat_dim)
        
        # Process landmarks: project (x, y) coordinates.
        landmark_features = self.landmark_proj(landmarks)  # (batch, T, num_landmarks, mobile_feat_dim)
        # Fuse visual features with landmark features.
        visual_features_expanded = visual_features.unsqueeze(2).expand_as(landmark_features)
        fused_features = visual_features_expanded + landmark_features  # (batch, T, num_landmarks, mobile_feat_dim)
        
        # Use self-adaptive adjacency matrix if external adj is not provided.
        if adj is None:
            adj = self.adaptive_adj
        
        spatial_features = []
        for t in range(T):
            x = fused_features[:, t, :, :]  # (batch, num_landmarks, mobile_feat_dim)
            x = F.relu(self.gcn1(x, adj))
            x = F.relu(self.gcn2(x, adj))
            x = torch.mean(x, dim=1)  # (batch, spatial_feature_dim)
            spatial_features.append(x)
        spatial_features = torch.stack(spatial_features, dim=1)  # (batch, T, spatial_feature_dim)
        temporal_input = spatial_features.transpose(1, 2)  # (batch, spatial_feature_dim, T)
        wavenet_out = F.relu(self.wavenet(temporal_input))  # (batch, spatial_feature_dim, T)
        temporal_features = torch.mean(wavenet_out, dim=2)    # (batch, spatial_feature_dim)
        logits = self.fc(temporal_features)                   # (batch, num_classes)
        return logits, fused_features

#############################################
#             Evaluation Function         #
#############################################

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC and F1 score computation

    with torch.no_grad():
        for frames, landmarks, labels in dataloader:
            frames = frames.to(device)           # (batch, T, C, H, W)
            landmarks = landmarks.to(device)     # (batch, T, 68, 2)
            labels = labels.to(device)
            # Let the model use its self-adaptive adjacency matrix.
            outputs, _ = model(frames, landmarks, adj=None)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    overall_acc = correct / total
    return overall_acc, all_preds, all_labels, all_probs

def plot_roc_curves(all_labels, all_probs, classes, save_path="roc_curve.png"):
    # Binarize ground truth.
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    n_classes = len(classes)
    labels_binarized = label_binarize(all_labels, classes=range(n_classes))
    
    plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size': 14})
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"ROC curve image saved to {save_path}")

#############################################
#            Main Evaluation              #
#############################################

def main():
    TEST_DIR = "preprocessed/test"
    NUM_FRAMES = 16
    BATCH_SIZE = 8

    test_dataset = PreprocessedDataset(TEST_DIR, num_frames=NUM_FRAMES)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FatigueDetectionModel(num_landmarks=68, spatial_feature_dim=128, num_classes=3, num_frames=NUM_FRAMES)
    model.to(device)
    
    model_path = f"best_fatigue_model_{NUM_FRAMES}.pth"
    if os.path.exists(model_path):
        print("Loading model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found!")
        return
    
    overall_acc, all_preds, all_labels, all_probs = evaluate(model, test_loader, device)
    print(f"Overall Test Accuracy: {overall_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Talking", "Yawning"], digits=4))
    
    # Per-class accuracy.
    class_counts = collections.Counter(all_labels)
    class_correct = {cls: 0 for cls in class_counts}
    for true, pred in zip(all_labels, all_preds):
        if true == pred:
            class_correct[true] += 1
    per_class_acc = {cls: class_correct[cls] / class_counts[cls] for cls in class_counts}
    print("Per-class Accuracy:")
    for cls, acc in per_class_acc.items():
        print(f"  Class {cls}: {acc:.4f}")
    
    # Macro F1 score.
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    print(f"Macro F1 Score: {f1_macro:.4f}")
    
    # Multi-class AUC-ROC.
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        print(f"Multi-class AUC-ROC (macro-average): {auc_roc:.4f}")
    except Exception as e:
        print("Error computing AUC-ROC:", e)
        auc_roc = None

    classes = ["Normal", "Talking", "Yawning"]
    plot_roc_curves(all_labels, all_probs, classes, save_path="roc_curve.png")

if __name__ == '__main__':
    main()