import os
from collections import defaultdict

def collect_filenames(data_dir):
    """
    Collects filenames from each class subdirectory within a given data split.
    
    :param data_dir: Base directory containing the split subdirectories.
    :param split_name: Name of the split ('train', 'val', or 'test').
    :return: Dictionary mapping class names to sets of filenames.
    """
    filenames_by_class = defaultdict(set)
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            filenames = {file for file in os.listdir(class_dir)}
            filenames_by_class[class_name].update(filenames)
    
    return filenames_by_class

def validate_data_splits(train_dir, val_dir, test_dir):
    """
    Validates that data splits (train, val, test) are mutually exclusive per class.
    
    :param data_dir: Base directory containing 'train', 'val', and 'test' subdirectories.
    """
    # Collect filenames for train, val, and test splits
    assert all([os.path.exists(split_dir) for split_dir in [train_dir, val_dir, test_dir]]), "One or more data splits do not exist."
    train_files = collect_filenames(train_dir)
    val_files = collect_filenames(val_dir)
    test_files = collect_filenames(test_dir)
    
    # Check for intersections between splits for each class
    for class_name in train_files.keys():
        train_val_intersection = train_files[class_name].intersection(val_files[class_name])
        train_test_intersection = train_files[class_name].intersection(test_files[class_name])
        val_test_intersection = val_files[class_name].intersection(test_files[class_name])
        
        if train_val_intersection:
            print(f'Overlap between train and val sets in class {class_name}: {train_val_intersection}')
        if train_test_intersection:
            print(f'Overlap between train and test sets in class {class_name}: {train_test_intersection}')
        if val_test_intersection:
            print(f'Overlap between val and test sets in class {class_name}: {val_test_intersection}')

        if not any([train_val_intersection, train_test_intersection, val_test_intersection]):
            #print(f'No overlap found in class {class_name}. Data splits are mutually exclusive.')
            pass
# Example usage:
train_dir = "/home/michael/Github/llava_test/mixed_train_sets/train_20_synth"
val_dir = "/home/michael/Github/llava_test/models/imagenet_100/val"
test_dir = "/home/michael/Github/llava_test/models/imagenet_100/test"
validate_data_splits(train_dir, val_dir, test_dir)
