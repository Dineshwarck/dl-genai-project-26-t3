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


class scSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Block.
    (Roy et al., IEEE TMI 2018).
    - cSE: Squeezes spatial context to recalibrate channel-wise feature dependencies (defects vs background).
    - sSE: Squeezes channel context via 1x1 conv to highlight defect spatial regions.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 4), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class SEResBlock(nn.Module):
    """
    Residual Convolutional Block with scSE attention.
    - Identity shortcut solves gradient decay across deep scratch architectures.
    - Double 3x3 conv with BatchNorm and ReLU.
    - scSE dynamically boosts defect features while suppressing steel surface grain.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

        self.scse = scSEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.scse(out)
        return self.relu(out + residual)


class ASPPBottleneck(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Bottleneck with Anisotropic Strip Convolutions.
    Specially engineered for elongated surface defects (scratches spanning hundreds of pixels horizontally):
    - Branch 0: 1x1 conv (local context)
    - Branch 1: 3x3 dilated conv (rate=2)
    - Branch 2: 3x3 dilated conv (rate=4)
    - Branch 3: Strip Convolutions (1x7 and 7x1) for horizontal scratch continuity across strip
    - Branch 4: Global Image Average Pooling (sheet-wide context)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out0 = self.b0(x)
        out1 = self.b1(x)
        out2 = self.b2(x)
        out3 = self.b3(x)
        out4 = nn.functional.interpolate(self.b4(x), size=(h, w), mode='bilinear', align_corners=True)
        cat = torch.cat([out0, out1, out2, out3, out4], dim=1)
        return self.project(cat)


class ResAttentionUNet(nn.Module):
    """
    Next-Generation Scratch Surface Inspection Architecture:
    - Residual SE-Blocks (identity shortcuts + scSE spatial/channel attention)
    - ASPP Bottleneck with Anisotropic Strip Convolutions (1x7, 7x1)
    - Additive Attention Gates on all 4 skip connections (filters clean-sheet reflection noise)
    - Multi-scale auxiliary classification head (Enc1 MaxPool + Enc2 MaxPool + Bottleneck AvgPool)
    - Optional Deep Supervision heads at Dec3 and Dec2 for accelerated scratch convergence
    Trained 100% from scratch with ZERO external pre-trained weights.
    """
    def __init__(self, in_channels=3, out_channels=4, deep_supervision=False):
        super().__init__()
        self.with_classifier = True
        self.with_attention = True
        self.deep_supervision = deep_supervision

        # --- Encoder (Downsampling) with Residual scSE Blocks ---
        self.enc1 = SEResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = SEResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = SEResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = SEResBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Multi-Scale ASPP Bottleneck ---
        self.bottleneck = ASPPBottleneck(512, 1024)

        # --- Multi-Scale Auxiliary Classification Head ---
        self.max_pool_enc1 = nn.AdaptiveMaxPool2d((1, 1))
        self.max_pool_enc2 = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool_b = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(64 + 128 + 1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, out_channels),
        )

        # --- Decoder (Upsampling) with Residual scSE Blocks ---
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = SEResBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = SEResBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = SEResBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = SEResBlock(128, 64)

        # --- Final Segmentation Head ---
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # --- Deep Supervision Heads (Optional) ---
        if deep_supervision:
            self.ds3 = nn.Conv2d(256, out_channels, kernel_size=1)
            self.ds2 = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x, return_cls=False):
        # 1. Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # 2. Multi-Scale ASPP Bottleneck
        b = self.bottleneck(p4)

        # 3. Auxiliary Classification Logits
        cls_logits = None
        if return_cls or self.training:
            feat_e1 = self.max_pool_enc1(e1).flatten(1)
            feat_e2 = self.max_pool_enc2(e2).flatten(1)
            feat_b = self.avg_pool_b(b).flatten(1)
            feat = torch.cat([feat_e1, feat_e2, feat_b], dim=1)
            cls_logits = self.cls_head(feat)

        # 4. Decoder with Attention Gates
        d4_up = self.upconv4(b)
        x4 = self.att4(g=d4_up, x=e4)
        d4 = self.dec4(torch.cat([d4_up, x4], dim=1))

        d3_up = self.upconv3(d4)
        x3 = self.att3(g=d3_up, x=e3)
        d3 = self.dec3(torch.cat([d3_up, x3], dim=1))

        d2_up = self.upconv2(d3)
        x2 = self.att2(g=d2_up, x=e2)
        d2 = self.dec2(torch.cat([d2_up, x2], dim=1))

        d1_up = self.upconv1(d2)
        x1 = self.att1(g=d1_up, x=e1)
        d1 = self.dec1(torch.cat([d1_up, x1], dim=1))

        # 5. Final Output
        mask_logits = self.final_conv(d1)

        if self.training and self.deep_supervision:
            ds3_out = nn.functional.interpolate(self.ds3(d3), size=x.shape[2:], mode='bilinear', align_corners=True)
            ds2_out = nn.functional.interpolate(self.ds2(d2), size=x.shape[2:], mode='bilinear', align_corners=True)
            return mask_logits, cls_logits, ds3_out, ds2_out

        if return_cls or (self.training and cls_logits is not None):
            return mask_logits, cls_logits
        return mask_logits

# Dedicated Scratch Classifier
class ChannelSE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(in_channels // reduction, 4), in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        return x * self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)


class SpatialSE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.conv(x))


class Classifier_scSE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.cse = ChannelSE(in_channels, reduction)
        self.sse = SpatialSE(in_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cse(x) + self.sse(x)


class ClassifierResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = Classifier_scSE(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.scse(out)
        return self.relu(out + res)


class SurfaceDefectClassifier(nn.Module):
    """
    Dedicated 100% Scratch Multi-Label Classifier for Industrial Steel Sheets.
    
    Architectural Highlights:
      1. 5-Stage Residual Backbone with Spatial & Channel Squeeze-and-Excitation (scSE).
      2. Dual Global Pooling: Concatenates AdaptiveAvgPool2d (broad textures) 
         and AdaptiveMaxPool2d (sharp 1-pixel scratch spikes).
      3. Regularized Dense Classification Head with Dropout(0.4) and BatchNorm1d.
      4. Outputs 4 independent logits (Defects A, B, C, D) without softmax constraint
         (allows multi-label co-occurrence).
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()

        # --- Stage 1 to 5: Feature Extraction ---
        # Input: (B, 3, 128, 800)
        self.stage1 = ClassifierResBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.stage2 = ClassifierResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage3 = ClassifierResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage4 = ClassifierResBlock(256, 384)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stage5 = ClassifierResBlock(384, 512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dual Global Pooling (Avg + Max) ---
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        pooled_dim = 512 * 2

        # --- Multi-Label Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.pool4(self.stage4(x))
        x = self.pool5(self.stage5(x))

        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        combined_feat = torch.cat([avg_feat, max_feat], dim=1)

        logits = self.classifier(combined_feat)
        return logits

# ======================================================================
# MODEL 3: Pretrained ResNet-34 U-Net (Transfer Learning & Fine-Tuning)
# ======================================================================

class ResNetDecoderBlock(nn.Module):
    """
    Symmetrical U-Net Decoder Block for ResNet backbones:
    1. Upsamples coarse latent features by 2x via ConvTranspose2d
    2. Concatenates with fine-grained encoder skip connection
    3. Refines features via Double Conv (two 3x3 convs + BatchNorm + ReLU)

    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against any 1-pixel rounding discrepancy
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = nn.functional.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PretrainedResNet34UNet(nn.Module):
    """
    Pretrained ResNet-34 U-Net for Industrial Steel Surface Defect Segmentation.

    Architectural Specifications:
    - Encoder: ImageNet-pretrained ResNet-34 (Torchvision)
    - Multi-Scale Skips:
        * Skip 0 (Stem Conv): (B, 64, 64, 400) - Fine spatial edges
        * Skip 1 (ResNet Layer 1): (B, 64, 32, 200) - Micro-textures
        * Skip 2 (ResNet Layer 2): (B, 128, 16, 100) - Surface patterns
        * Skip 3 (ResNet Layer 3): (B, 256, 8, 50) - Defect geometries
        * Latent (ResNet Layer 4): (B, 512, 4, 25) - High-level semantics
    - Multi-Task Auxiliary Classification Head:
        * Global Pooling + MLP predicting 4 defect presence logits
    - Symmetrical Decoder:
        * Decodes (4, 25) back to full resolution (128, 800) with 4 defect channels
    - Built-in Progressive Fine-Tuning controls:
        * freeze_encoder() / unfreeze_encoder()
        * get_parameter_groups() for differential learning rates
    """    
    
    def __init__(self, in_channels: int = 3, out_channels: int = 4, pretrained: bool = True, with_classifier: bool = True):
        super().__init__()
        self.with_classifier = with_classifier
        self.with_attention = False

        # 1. Load Pretrained ResNet-34 Backbone ---
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            base = resnet34(weights=weights)
        except (ImportError, AttributeError):
            import torchvision.models as models
            base = models.resnet34(pretrained=pretrained)
        
        # Encoder Stem: (B, 3, 128, 800) -> (B, 64, 64, 400)
        if in_channels == 3:
            self.conv1 = base.conv1
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool # (B, 64, 64, 400) -> (B, 64, 32, 200)

        # Encoder Residual Stages
        self.layer1 = base.layer1 # Output: (B, 64, 32, 200)
        self.layer2 = base.layer2 # Output: (B, 128, 16, 100)
        self.layer3 = base.layer3 # Output: (B, 256, 8, 50)
        self.layer4 = base.layer4 # Output: (B, 512, 4, 25)

        # --- 2. Multi-Task Auxiliary Classification Head ---
        if with_classifier:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))    
            self.cls_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256, out_channels)
            )
        
        # --- 3. Symmetrical U-Net Decoder ---
        # Dec 4: (4, 25) -> (8, 50), receives Skip 3 (256 ch)
        self.dec4 = ResNetDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        # Dec 3: (8, 50) -> (16, 100), receives Skip 2 (128 ch)
        self.dec3 = ResNetDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        # Dec 2: (16, 100) -> (32, 200), receives Skip 1 (64 ch)
        self.dec2 = ResNetDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        # Dec 1: (32, 300) -> (64, 400), receives Skip 0 (64 ch)
        self.dec1 = ResNetDecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Head: (64, 400) -> (128, 800) -> 4 Defect Classes
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x:torch.Tensor, return_cls: bool = False):
        # --- Encoder Forward Pass ---
        # Stem
        s0 = self.relu(self.bn1(self.conv1(x))) # Skip 0: (B, 64, 64, 400)
        x_p = self.maxpool(s0) # (B,64,32,200)

        # Stages
        s1 = self.layer1(x_p) # Skip 1: (B, 64, 32, 200)
        s2 = self.layer2(s1) # Skip 2: (B, 128, 16, 100)
        s3 = self.layer3(s2) # Skip 3: (B, 256, 8, 50)
        b = self.layer4(s3) # Latent: (B, 512, 4, 25)

        # Optional Auxiliary Classifier
        cls_logits = None
        if self.with_classifier:
            pooled = self.avg_pool(b).flatten(1)
            cls_logits = self.cls_head(pooled)
            
        # --- Decoder Forward Pass ---
        d4 = self.dec4(b, s3) # -> (B, 256, 8, 50)
        d3 = self.dec3(d4, s2) # -> (B, 128, 16, 100)
        d2 = self.dec2(d3, s1) # -> (B, 64, 32, 200)
        d1 = self.dec1(d2, s0) # -> (B, 64, 64, 400)
        
        mask_logits = self.final_conv(d1) # -> (B, 4, 128, 800)
        
        if return_cls or (self.training and cls_logits is not None):
            return mask_logits, cls_logits
        return mask_logits

    def freeze_encoder(self):
        """
        Freezes pre-trained ResNet-34 encoder weights for Phase 1 warmup

        """
        for name, module in [('conv1', self.conv1), ('bn1', self.bn1), ('layer1', self.layer1), ('layer2', self.layer2), ('layer3', self.layer3), ('layer4', self.layer4)]:
            for param in module.parameters():
                param.requires_grad = False
        
        print("[PretrainedResNet34UNet] Encoder successfully frozen for Phase 1 warmup.")
        
    def unfreeze_encoder(self):
        """
        Unfreezes pre-trained ResNet-34 encoder weights for phase 2 fine-tuning.

        """
        for name, module in [('conv1', self.conv1), ('bn1', self.bn1), ('layer1', self.layer1), ('layer2', self.layer2), ('layer3', self.layer3), ('layer4', self.layer4)]:
            for param in module.parameters():
                param.requires_grad = True
            
        print("[PretrainedResNet34UNet] Encoder successfully unfrozen for Phase 2 fine-tuning.")

    def get_paramter_groups(self, lr_encoder: float = 1e-5, lr_decoder: float = 1e-4, weight_decay: float = 1e-4):
        """
        Constructs parameter groups with discriminative learning rates:
         - Pretrained Encoder: Low LR (1e-5) to prevent catastrophic forgetting
         - Custom Decoder & Heads: Normal LR (1e-4) for rapid task convergence

        """

        encoder_params = (
            list(self.conv1.parameters()) + 
            list(self.bn1.parameters()) + 
            list(self.layer1.parameters()) + 
            list(self.layer2.parameters()) + 
            list(self.layer3.parameters()) + 
            list(self.layer4.parameters())
        )

        decoder_params = (
            list(self.dec4.parameters()) + 
            list(self.dec3.parameters()) + 
            list(self.dec2.parameters()) + 
            list(self.dec1.parameters()) + 
            list(self.final_conv.parameters()) +
            (list(self.cls_head.parameters()) if self.with_classifier else [])
        )

        return [
            {"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": lr_decoder, "weight_decay": weight_decay}
        ]

    # Alias for compatibility
    def get_parameter_groups(self, lr_encoder: float = 1e-5, lr_decoder: float = 1e-4, weight_decay: float = 1e-4):
        return self.get_paramter_groups(lr_encoder=lr_encoder, lr_decoder=lr_decoder, weight_decay=weight_decay)