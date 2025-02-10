
from torchvision import  transforms


def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        # normalize_transform(pretrained)
    ])
    return train_transform
# Validation transforms
T = transforms.RandomChoice([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(0.85,0.85,0.85,0.5),
        transforms.RandomInvert(0.3),
        transforms.GaussianBlur(7),
        transforms.RandomAdjustSharpness(0.33),
        transforms.Grayscale(3),
        transforms.RandomRotation(10)
        ])
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        # normalize_transform(pretrained)
    ])
    return valid_transform
# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize
normalize_pretrained = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
