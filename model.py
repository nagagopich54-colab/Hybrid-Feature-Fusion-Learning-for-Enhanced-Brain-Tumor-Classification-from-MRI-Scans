# Imports
import torch
import torch.nn as nn
import timm
from transformers import BeitModel

class Hybrid_BEiT_EffNetV2M(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # BEiT Transformer branch
        self.beit = BeitModel.from_pretrained("microsoft/beit-base-patch16-224")
        beit_dim = self.beit.config.hidden_size

        # EfficientNetV2-M CNN branch
        self.effnet = timm.create_model("tf_efficientnetv2_m_in21k", pretrained=True)
        eff_dim = self.effnet.num_features

        # Remove classifier head
        self.effnet.classifier = nn.Identity()

        # Fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(beit_dim + eff_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        eff_feat = self.effnet(x)
        beit_feat = self.beit(pixel_values=x).pooler_output
        return self.classifier(torch.cat([eff_feat, beit_feat], dim=1))
