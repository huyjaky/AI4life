import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from urllib.request import urlretrieve
import os


class CustomVDModel(nn.Module):
    def __init__(self, num_classes):
        super(CustomVDModel, self).__init__()
        self.base_model = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
        )

        # freeze features extractor's params
        # for param in self.base_model.parameters():
        #   param.requires_grad = False
        self.in_features = self.base_model.classifier[0].in_features

        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        self.global_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        self.human_classfier = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1280),
            nn.ReLU(),
            nn.Linear(in_features=1280, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )
        self.violence_classfier = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1280),
            nn.ReLU(),
            nn.Linear(in_features=1280, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_pool(x)
        human_output = self.human_classfier(x)
        violence_output = self.violence_classfier(x)
        return violence_output, human_output


model = CustomVDModel(num_classes=2)
# Explicitly map to CPU when loading
model.load_state_dict(
    torch.load("./archive/VDModel_2.pth", map_location=torch.device("cpu"))
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

model.to(device)
model.eval()

def detect_violence(image_model=""):

    img = Image.open(image_model).convert("RGB")

    # plt.imshow(img)
    # plt.show()

    # img = Image.open('/content/data/human_dataset/non-human/1090.jpg').convert('RGB')

    # plt.imshow(img)
    # plt.show()

    preprocess = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tensor_img = preprocess(img)

    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        violence_output, human_output = model(tensor_img)

        violence_output = nn.functional.softmax(violence_output, dim=1)
        human_output = nn.functional.softmax(human_output, dim=1)

        violence_pred = torch.argmax(violence_output, 1).item()
        human_pred = torch.argmax(human_output, 1).item()

    print(f"Violence Prediction: {violence_pred}, Human Prediction: {human_pred}")
    return {"violence_pred": violence_pred, "human_pred": human_pred}


# url = "https://suckhoedoisong.qltns.mediacdn.vn/zoom/600_315/324455921873985536/2022/5/31/danh-nhau-16539552795601460283475-0-58-270-490-crop-16539552889991400211904.jpeg"
# detect_violence(model=model, img_url=url, file_name="test8")
