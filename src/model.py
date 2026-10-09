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
    

class ScratchUNet(nn.Module):
    def __init__(self,in_channels=3,out_channels=4):
        super().__init__()

        # --- Encoder (Downsampling) ---
        self.enc1 = DoubleConv(in_channels,64)  
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

        self.enc2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

        # --- Bottleneck (Deepest Layer) ---
        self.bottleneck = DoubleConv(128,256)

        # --- Decoder (Upsampling) ---
        # ConvTranspose2d doubles the height and width, but halves the channels

        self.upconv2 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        # After upsampling, we concatenate with the encoder features (128 + 128 = 256)
        self.dec2 = DoubleConv(256,128)

        self.upconv1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)

        # Concatenate again (64+64 = 128)
        self.dec1 = DoubleConv(128,64)

        # --- FINAL LAYER ---
        # 1x1 Convolution to map 64 features to our 4 defects classes
        self.final_conv = nn.Conv2d(64,out_channels,kernel_size=1)

    def forward(self,x):
        # 1. Pass through Encoder 1
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        # 2. Pass through Encoder 2
        e2 = self.enc2(p1)
        p2 = self.pool2(e2) 

        # 3. Pass through Bottleneck
        b = self.bottleneck(p2)

        # 4. Decoder 2: Upsample bottleneck, concatenate with e2, and pass through dec2

        d2 = self.upconv2(b)
        # Concatenation: PyTorch tensor concatenation along the channel dimension (dim=1)
        d2_cat = torch.cat((d2,e2),dim=1)
        d2_out = self.dec2(d2_cat)
 
        # 5. Decoder 1: Upsample d2_out, concatenate with e1, and pass through dec1
        d1 = self.upconv1(d2_out)
        d1_cat = torch.cat((d1,e1),dim=1)
        d1_out = self.dec1(d1_cat)

        # 6. Final Convolution
        out = self.final_conv(d1_out)

        return out