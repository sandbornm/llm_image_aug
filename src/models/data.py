import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import random

from torchvision.datasets import ImageFolder

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends torchvision.datasets.ImageFolder"""
    
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # image file path
        path = self.imgs[index][0]
        # preserve image path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def create_deterministic_symlinked_dataset(source_dir, target_dir, min_images=400, num_select=400, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    class_counter = 0

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)

    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        images = sorted(os.listdir(class_dir))
        if len(images) < min_images:
            continue
        
        selected_images = images[:num_select]
        
        # split indices
        num_train = int(num_select * train_ratio)
        num_val = int(num_select * val_ratio)
        
        train_images = selected_images[:num_train]
        val_images = selected_images[num_train:num_train + num_val]
        test_images = selected_images[num_train + num_val:]
        
        def symlink_images(images, split):
            split_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img_name in images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(split_dir, img_name)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        
        symlink_images(train_images, 'train')
        symlink_images(val_images, 'val')
        symlink_images(test_images, 'test')
        
        class_counter += 1

    print(f"Total number of classes processed: {class_counter}")
    return class_counter


def get_transforms(use_224=True):
    base_size = 224 if use_224 else 256
    resize_size = 224 if use_224 else 256  # validation resizing

    resize_tform = transforms.Resize((base_size, base_size)) if use_224 else transforms.Lambda(lambda x: x)

    train_transforms = transforms.Compose([
        resize_tform,
        transforms.RandomResizedCrop(base_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        resize_tform,
        transforms.Resize(resize_size),
        transforms.CenterCrop(base_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms


def create_dataloaders(target_dir, train_transforms, val_transforms, batch_size=16, num_workers=4):
    """
    Creates PyTorch DataLoader instances for training, validation, and testing datasets.

    Parameters:
    - target_dir: Base directory containing 'train', 'val', and 'test' subdirectories.
    - train_transforms: Transformations to apply to the training dataset.
    - val_transforms: Transformations to apply to the validation and test datasets.
    - batch_size: Batch size for the DataLoader instances.
    - num_workers: Number of subprocesses to use for data loading.

    Returns:
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - test_loader: DataLoader for the test dataset.
    """
    
    train_dataset = datasets.ImageFolder(os.path.join(target_dir, 'train'), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(target_dir, 'val'), transform=val_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(target_dir, 'test'), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_val_test_loaders(base_dir: str, batch_size: int=16):
    _, val_transforms = get_transforms()
    val_dataset = datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=val_transforms)
    test_dataset = datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return val_loader, test_loader

def get_train_loader(train_dir: str, batch_size: int=16):
    train_transforms, _ = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader


def create_augmented_training_sets(real_train_dir, synth_dir, output_base_dir, num_images_per_class, seed=42):
    """
    Create augmented training sets by adding specific numbers of synthetic images to the training set.
    
    :param real_train_dir: Directory with the real training images.
    :param synth_dir: Directory with the synthetic images, containing class-named subdirectories.
    :param output_base_dir: Base directory to store the augmented training sets.
    :param num_images_per_class: List specifying numbers of synthetic images to add per class.
    :param seed: Random seed for consistent selection of images.
    """
    random.seed(seed)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    class_names = os.listdir(real_train_dir)
    
    for num_images in num_images_per_class:
        new_train_dir_name = f'train_{num_images}synth'
        new_train_dir = os.path.join(output_base_dir, new_train_dir_name)
        os.makedirs(new_train_dir, exist_ok=True)
        
        for class_name in class_names:
            real_class_dir = os.path.join(real_train_dir, class_name)
            synth_class_dir = os.path.join(synth_dir, class_name)
            new_class_dir = os.path.join(new_train_dir, class_name)
            os.makedirs(new_class_dir, exist_ok=True)
            
            real_images = sorted(os.listdir(real_class_dir))
            synth_images = sorted(os.listdir(synth_class_dir))[:num_images]
            
            random.shuffle(real_images)
            images_to_keep = real_images[num_images:]  # Keep real images after replacement
            
            # Symlink synthetic images to the new directory
            for img_name in synth_images:
                src = os.path.join(synth_class_dir, img_name)
                dst = os.path.join(new_class_dir, img_name)
                os.symlink(src, dst)
            
            # Symlink the real images that are kept
            for img_name in images_to_keep:
                src = os.path.join(real_class_dir, img_name)
                dst = os.path.join(new_class_dir, img_name)
                os.symlink(src, dst)
    
def validate_synth_sets(synth_base_dir: str, num_images: int=320):
        for sdir in os.listdir(synth_base_dir):
            sdir_path = os.path.join(synth_base_dir, sdir)
            for class_dir in os.listdir(sdir_path):
                class_path = os.path.join(sdir_path, class_dir)
                assert len(os.listdir(class_path)) == num_images, f"Expected {num_images} images in {class_path}, found {len(os.listdir(class_path))}"
        
        
if __name__  == "__main__":
    real_data_dir = '/home/michael/Github/llava_test/models/imagenet_100/train'
    synthetic_data_dir = '/home/michael/Github/llava_test/models/imagenet_100_synthetic'
    output_base_dir = '/home/michael/Github/llava_test/mixed_train_sets'
    
    regen_synth = False

    if regen_synth:
        num_images_per_class = [20, 40, 80, 100]
        create_augmented_training_sets(real_data_dir, synthetic_data_dir, output_base_dir, num_images_per_class)
        
    validate_synth_sets(output_base_dir)
