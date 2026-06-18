# Imports
import torch, torch.nn as nn
import timm, cv2, numpy as np, matplotlib.pyplot as plt
from torchvision import transforms
from transformers import BeitModel
from PIL import Image

# Device
device = "cpu"

# Input image paths
image_paths = [
    "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train/glioma/brisc2025_train_00007_gl_ax_t1.jpg",
    "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train/meningioma/brisc2025_train_01155_me_ax_t1.jpg",
    "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train/meningioma/brisc2025_train_01162_me_ax_t1.jpg",
    "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/test/pituitary/brisc2025_test_00725_pi_ax_t1.jpg",
    "/kaggle/input/datasets/briscdataset/brisc2025/brisc2025/classification_task/train/meningioma/brisc2025_train_01174_me_ax_t1.jpg"
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load images
imgs, originals = [], []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    originals.append(np.array(img.resize((224,224))))
    imgs.append(transform(img))

# Stack images into batch
x = torch.stack(imgs)

# Hybrid model (same as training)
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNetV2-M branch
        self.effnet = timm.create_model("tf_efficientnetv2_m_in21k", pretrained=True)
        self.effnet.classifier = nn.Identity()

        # BEiT Transformer branch
        self.beit = BeitModel.from_pretrained(
            "microsoft/beit-base-patch16-224",
            attn_implementation="eager"
        )

        # Final classifier
        self.fc = nn.Linear(self.effnet.num_features + self.beit.config.hidden_size, 3)

        # For Grad-CAM++
        self.activations, self.gradients = None, None

        # Target layer for Grad-CAM
        target_layer = self.effnet.blocks[-3]

        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,"activations",o)
        )
        target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,"gradients",go[0])
        )

    def forward(self, x):
        # CNN features
        eff = self.effnet(x)

        # Transformer features + attention
        beit_out = self.beit(
            pixel_values=x,
            output_attentions=True,
            return_dict=True
        )

        # Store attention maps
        self.attentions = beit_out.attentions

        # Feature fusion
        fused = torch.cat([eff, beit_out.pooler_output], dim=1)
        return self.fc(fused)

# Load model
model = HybridModel().to(device)
model.eval()

# Forward pass
with torch.no_grad():
    outputs = model(x)

# Predictions
probs = torch.softmax(outputs, dim=1)
preds = probs.argmax(dim=1)

# Normalize function
def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

# Overlay heatmap on image
def overlay(img, hm):
    hm = cv2.resize(hm, (224,224))
    hm = norm01(hm)
    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.6, hm, 0.4, 0)

# Safe indexing for batch
def safe_idx(t, i):
    return t[i] if t.shape[0] > 1 else t[0]

# Feature extractor for feature maps
feat_model = timm.create_model(
    "tf_efficientnetv2_m_in21k",
    pretrained=True,
    features_only=True
).to(device)

with torch.no_grad():
    feats = feat_model(x)

# Convert feature map to heatmap
def fmap_to_hm(f):
    hm = torch.max(f.detach(), dim=0)[0].numpy()
    return norm01(cv2.resize(hm,(224,224)))

# Attention rollout computation
def rollout(attns):
    r = torch.eye(attns[0].size(-1))
    for a in attns:
        a = a.mean(dim=1)
        a = a + torch.eye(a.size(-1))
        a = a / a.sum(dim=-1, keepdim=True)
        r = torch.matmul(a, r)
    return r

roll = rollout(model.attentions)

# Grad-CAM++
def gradcam_pp(model, inp, cls):
    model.zero_grad()
    out = model(inp)
    out[0, cls].backward()

    g = model.gradients[0]
    a = model.activations[0]

    g2 = g**2
    g3 = g**3
    sum_g = torch.sum(g, dim=(1,2), keepdim=True)

    alpha = g2 / (2*g2 + sum_g * g3 + 1e-8)
    weights = torch.sum(alpha * torch.relu(g), dim=(1,2))

    cam = torch.sum(weights[:,None,None] * a, dim=0)
    cam = torch.relu(cam).detach().numpy()

    return norm01(cam)

# Grad-CAM++ outputs
plt.figure(figsize=(10,10))
for i in range(5):
    cam = gradcam_pp(model, x[i].unsqueeze(0), preds[i].item())

    plt.subplot(5,2,2*i+1)
    plt.imshow(originals[i]); plt.title("Input"); plt.axis('off')

    plt.subplot(5,2,2*i+2)
    plt.imshow(overlay(originals[i], cam)); plt.title("Grad-CAM++"); plt.axis('off')

plt.suptitle("Grad-CAM++", fontsize=14)
plt.tight_layout(); plt.show()

# Attention Maps outputs
plt.figure(figsize=(14,10))
layers = [0,3,7,11]

for i in range(3):
    plt.subplot(3,6,i*6+1)
    plt.imshow(originals[i]); plt.title("Input"); plt.axis('off')

    r = safe_idx(roll,i)[0,1:]
    size = int(np.sqrt(r.shape[0]))
    r_map = norm01(r.reshape(size,size).detach().numpy())

    plt.subplot(3,6,i*6+2)
    plt.imshow(overlay(originals[i], r_map)); plt.title("Rollout"); plt.axis('off')

    for j,l in enumerate(layers):
        attn = safe_idx(model.attentions[l],i)
        attn = attn.mean(dim=0)[0,1:]
        size = int(np.sqrt(attn.shape[0]))
        attn = attn.reshape(size,size).detach().numpy()

        plt.subplot(3,6,i*6+3+j)
        plt.imshow(overlay(originals[i], attn))
        plt.title(f"L{l+1}"); plt.axis('off')

plt.suptitle("Attention Analysis", fontsize=14)
plt.tight_layout(); plt.show()

#Feature Maps outputs 
plt.figure(figsize=(12,6))
for i in range(2):
    plt.subplot(2,6,i*6+1)
    plt.imshow(originals[i]); plt.title("Input"); plt.axis('off')

    for j in range(5):
        fmap = fmap_to_hm(feats[j][i])
        plt.subplot(2,6,i*6+2+j)
        plt.imshow(overlay(originals[i], fmap))
        plt.title(f"L{j+1}"); plt.axis('off')

plt.suptitle("Feature Maps", fontsize=14)
plt.tight_layout(); plt.show()
