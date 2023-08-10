class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count
        # batch normalise input
        self.bn1 = nn.BatchNorm2d(self.input_shape.channels)
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv1)
        # batch normalise conv1
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.initialise_layer(self.conv2)
        # batch normalise conv2
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(4096, 1024)
        self.initialise_layer(self.fc1)
        # batch normalise fc1
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.initialise_layer(self.fc2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # batch normalise input
        x = self.bn1(images)
        x = self.conv1(images)
        # batch normalise X
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        # batch normalise X
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        # batch normalise X
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)