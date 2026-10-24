import torch
import torch.nn as nn
import torch.nn.functional as F


# Attention Gate module
class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionGate, self).__init__()

        # 1x1 convolution to reduce channels
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # x: skip connection feature map
        # g: gating signal (from decoder)

        # Apply 1x1 convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Add the features
        combined = F.relu(g1 + x1)

        # Apply a sigmoid activation to get the attention weights
        psi = self.sigmoid(self.psi(combined))

        # Apply the attention weights to the input feature map
        return x * psi


# DoubleConv block for U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# U-Net with Attention Gates
class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.center = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)  # Attention gate at the skip connection
        self.decoder4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.decoder3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.decoder2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Bottleneck (Center)
        center = self.center(p4)

        # Decoder with attention gates
        up4 = self.up4(center)
        att4 = self.att4(e4, up4)
        merge4 = torch.cat([up4, att4], 1)
        d4 = self.decoder4(merge4)

        up3 = self.up3(d4)
        up3_resized = F.interpolate(up3, size=e3.size()[2:], mode='bilinear', align_corners=False)
        att3 = self.att3(e3, up3_resized)
        merge3 = torch.cat([up3_resized, att3], 1)
        d3 = self.decoder3(merge3)

        up2 = self.up2(d3)
        up2_resized = F.interpolate(up2, size=e2.size()[2:], mode='bilinear', align_corners=False)
        att2 = self.att2(e2, up2_resized)
        merge2 = torch.cat([up2_resized, att2], 1)
        d2 = self.decoder2(merge2)

        up1 = self.up1(d2)
        up1_resized = F.interpolate(up1, size=e1.size()[2:], mode='bilinear', align_corners=False)
        att1 = self.att1(e1, up1_resized)
        merge1 = torch.cat([up1_resized, att1], 1)
        d1 = self.decoder1(merge1)

        # Final output
        out = self.final_conv(d1)
        return out


# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = AttentionUNet(in_channels=3, out_channels=1)

    # Create a random input tensor (e.g., for grayscale images 256x256)
    x = torch.randn(1, 3, 256, 256)

    # Forward pass through the model
    output = model(x)

    # Output shape should be [1, 1, 256, 256] for segmentation (batch_size, channels, height, width)
    print(output.shape)
