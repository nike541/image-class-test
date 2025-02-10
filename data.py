
from transforms import *

from torchvision import datasets
from torch.utils.data import DataLoader


def get_datasets(TRAIN_DIR,VAL_DIR,IMAGE_SIZE,pretrained):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset_train = datasets.ImageFolder(
        TRAIN_DIR, 
        transform=(get_train_transform(IMAGE_SIZE, pretrained))
    )
    dataset_valid = datasets.ImageFolder(
        VAL_DIR, 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    # dataset_size = len(dataset)
    # # Calculate the validation dataset size.
    # valid_size = int(VALID_SPLIT*dataset_size)
    # # Radomize the data indices.
    # indices = torch.randperm(len(dataset)).tolist()
    # # Training and validation sets.
    # dataset_train = Subset(dataset, indices[:-valid_size])
    # dataset_valid = Subset(dataset_test, indices[-valid_size:])
    
    return dataset_train, dataset_valid, dataset_train.classes

from torchsampler import ImbalancedDatasetSampler
def get_data_loaders(BATCH_SIZE,NUM_WORKERS,dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, 
        
        batch_size=BATCH_SIZE, 
        sampler=ImbalancedDatasetSampler(dataset_train),
        shuffle=False, num_workers=NUM_WORKERS,drop_last=True
    )
    valid_loader = DataLoader(
        dataset_valid,
        sampler=ImbalancedDatasetSampler(dataset_valid),
        batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS,drop_last=True
    )
    return train_loader, valid_loader 

