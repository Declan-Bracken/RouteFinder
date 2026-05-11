import torch.nn as nn
import open_clip
import torch

def build_model(device, dropout = 0.2):
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1)).to(device)
    return model, head

@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    all_scores, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = model.encode_image(imgs).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        scores = torch.sigmoid(head(feats).squeeze(1))
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.tolist())
    return all_scores, all_labels

def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    return model, head, ckpt

