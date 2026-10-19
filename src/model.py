import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        """
        A block consisting of two consecutive Convolutional layers,
        each followed by a BatchNorm and a ReLU activation.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)
    

class AttentionBlock(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 2018).
    Gates low-level encoder skip features x using coarse-level decoder features g,
    filtering out background surface reflections while preserving genuine defect activations.
    Trained 100% from scratch with no pre-trained weights.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ScratchUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, dropout_rate=0.3, with_classifier=True, with_attention=False):
        super().__init__()
        self.with_classifier = with_classifier
        self.with_attention = with_attention

        # --- Encoder (Downsampling) ---
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck (Deepest Layer) & Regularization ---
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # --- Multi-Scale MaxPool Auxiliary Classification Head ---
        # 1. Enc1 (64ch) & Enc2 (128ch): MaxPool captures peak activations of 1-pixel scratches without dilution!
        # 2. Bottleneck (1024ch): AvgPool captures global texture/area deformation!
        if with_classifier:
            self.max_pool_enc1 = nn.AdaptiveMaxPool2d((1, 1))
            self.max_pool_enc2 = nn.AdaptiveMaxPool2d((1, 1))
            self.avg_pool_b = nn.AdaptiveAvgPool2d((1, 1))
            # 64 + 128 + 1024 = 1216 channels
            self.cls_head = nn.Sequential(
                nn.Linear(1216, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, out_channels),
            )

        # --- Decoder (Upsampling) ---
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # --- Additive Attention Gates (Optional, Trained 100% from Scratch) ---
        if with_attention:
            self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
            self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
            self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
            self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        # --- FINAL LAYER ---
        # 1x1 Convolution to map 64 features to out_channels (e.g. 4 defect classes)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, return_cls=False):
        # 1. Encoder stages
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # 2. Bottleneck with Dropout
        b = self.bottleneck(p4)
        b = self.dropout(b)

        # 3. Auxiliary Classification Logits (Multi-Scale Fusion)
        cls_logits = None
        if self.with_classifier and (return_cls or self.training):
            feat_e1 = self.max_pool_enc1(e1).flatten(1)
            feat_e2 = self.max_pool_enc2(e2).flatten(1)
            feat_b = self.avg_pool_b(b).flatten(1)
            feat = torch.cat([feat_e1, feat_e2, feat_b], dim=1)
            cls_logits = self.cls_head(feat)

        # 4. Decoder stages with skip connections (gated by Attention if enabled)
        d4 = self.upconv4(b)
        x4 = self.att4(g=d4, x=e4) if self.with_attention else e4
        d4_cat = torch.cat((d4, x4), dim=1)
        d4_out = self.dec4(d4_cat)

        d3 = self.upconv3(d4_out)
        x3 = self.att3(g=d3, x=e3) if self.with_attention else e3
        d3_cat = torch.cat((d3, x3), dim=1)
        d3_out = self.dec3(d3_cat)

        d2 = self.upconv2(d3_out)
        x2 = self.att2(g=d2, x=e2) if self.with_attention else e2
        d2_cat = torch.cat((d2, x2), dim=1)
        d2_out = self.dec2(d2_cat)

        d1 = self.upconv1(d2_out)
        x1 = self.att1(g=d1, x=e1) if self.with_attention else e1
        d1_cat = torch.cat((d1, x1), dim=1)
        d1_out = self.dec1(d1_cat)

        # 5. Final 1x1 convolution
        mask_logits = self.final_conv(d1_out)

        if return_cls or (self.training and cls_logits is not None):
            return mask_logits, cls_logits
        return mask_logits