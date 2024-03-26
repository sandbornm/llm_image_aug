import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os

def plot_confusion_matrix(cm, model_name, weights, num_classes, class_names, test_set_size):
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Optional: Set the diagonal to zero if you want to focus on misclassifications.

    plt.figure(figsize=(20, 20))  # You may need an even larger figure size or use a vector format for the image.
    sns.set(font_scale=1.4)  # Adjust this based on the actual size of your matrix and figure.
    hm = sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True,
                     xticklabels=class_names, yticklabels=class_names)
    for i in range(cm_normalized.shape[0]):
        hm.text(i + 0.5, i + 0.5, f'{cm_normalized[i, i]:.2%}', 
                fontdict={'fontsize':8 if cm_normalized[i, i] != 0 else 0}, 
                ha='center', va='center', color='black')
    hm.set_xlabel('Predicted Labels', fontsize=18)
    hm.set_ylabel('True Labels', fontsize=18)
    hm.set_title(f'Model: {model_name} weights: {weights} ({num_classes} classes) test size: {test_set_size}', fontsize=22)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'/home/michael/Github/llm-img-aug/data/figs/{model_name}_{weights}.png', dpi=300)


# def plot_test_performance(test_perf_dir):


    
#     file_lst = glob.glob(test_perf_dir + '/*.json')
#     for file in file_lst:
#         model_name = file.split('/')[-1].split('_')[0]
#         weights = file.split('/')[-1].split('_')[1]
#         data_set = "baseline" if "baseline" in file else file.split('_train_')[1].split('.')[0]
#         with open(file, 'r') as f:
#             data = json.load(f)
#         test_acc = data['avg_test_acc']
    
def plot_test_performance(test_perf_dir):
    # Collect data in a structured form: { (model_name, weights) : {data_set: test_acc}}
    performance_data = {}

    # List all JSON files in the directory
    file_lst = glob.glob(os.path.join(test_perf_dir, '*.json'))
    for file in file_lst:
        # Extracting details from the filename
        filename = os.path.basename(file)
        model_name, weights = filename.split('_')[:2]
        data_set = "baseline" if "baseline" in file else filename.split('_train_')[1].split('.')[0]
        
        # Read the JSON file and extract the average test accuracy
        with open(file, 'r') as f:
            data = json.load(f)
        test_acc = data['avg_test_acc']
        
        # Add to our data structure
        performance_data.setdefault((model_name, weights), {})[data_set] = test_acc
    
    # Sort the datasets to ensure consistent order
    all_data_sets = sorted(set(ds for _, ds_dict in performance_data.items() for ds in ds_dict))

    # Prepare the plot
    plt.figure(figsize=(10, 6))
    for (model_name, weights), data_set_dict in performance_data.items():
        # Sort the test accuracies by dataset name
        test_accs = [data_set_dict.get(ds, None) for ds in all_data_sets]
        
        # Plot a line for each model
        plt.plot(all_data_sets, test_accs, marker='o', label=f'{model_name} {weights}')
    
    # Labeling the plot
    plt.xlabel('Dataset')
    plt.ylabel('Test Accuracy')
    plt.title('Test Performance by Model and Dataset')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit everything
    
    # Show the plot
    plt.show()
    

plot_test_performance("/home/michael/Github/llm-img-aug/data/test_performance")