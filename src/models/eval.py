import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
import os
from model import get_model
from data import get_transforms, ImageFolderWithPaths
from sklearn.metrics import confusion_matrix
from plot import plot_confusion_matrix


def evaluate_model(model, test_loader, device, output_path):
    model.to(device)
    model.eval()
    mispredictions = {}
    all_predictions = []
    all_true_labels = []
    correct_predictions = 0
    total_samples = 0

    class_names = test_loader.dataset.classes
    test_set_size = len(test_loader.dataset)

    with torch.no_grad():
        for inputs, labels, paths in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            confidences, predictions = torch.max(F.softmax(outputs, dim=1), 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
            
            top5_confidences, top5_preds = torch.topk(F.softmax(outputs, dim=1), 5, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            for i, (pred, true) in enumerate(zip(predictions, labels)):
                if pred != true:
                    mispredictions[paths[i]] = {
                        "Predicted": test_loader.dataset.classes[pred.item()],
                        "Actual": test_loader.dataset.classes[true.item()],
                        "Confidence": confidences[i].item(),
                        "top5 labels": [test_loader.dataset.classes[idx] for idx in top5_preds[i].tolist()],
                        "top5 confidences": top5_confidences[i].tolist()
                    }

    print(f"Total mispredictions for model {model_name}: {len(mispredictions)}/{len(test_loader.dataset)}")
    average_accuracy = (correct_predictions / total_samples) * 100
    print(f"Average Test Accuracy: {average_accuracy:.2f}%")

    cm = confusion_matrix(all_true_labels, all_predictions)
    
    evaluation_summary = {
        'avg_test_acc': average_accuracy,
        'mispredictions': mispredictions,
        'num_mispredictions': len(mispredictions),
        'confusion_matrix': cm.tolist()
    }

    if not os.path.exists(output_path):
        try:
            with open(output_path, 'w') as f:
                json.dump(evaluation_summary, f, indent=4)
        except IOError as e:
            print(f"Failed to save evaluation summary to {output_path}. Error: {e}")

        # plot cm from figures
        #plot_confusion_matrix(cm, model_name, model_weights, len(test_loader.dataset.classes), class_names, test_set_size)



if __name__ == '__main__':
    
    MODEL_DIR = os.path.join('/home/michael/Github/llm-img-aug/models')
    print(MODEL_DIR)
    
    test_dir = "/home/michael/Github/llava_test/models/imagenet_100"
    outdir = "/home/michael/Github/llm-img-aug/data/test_performance"
    
    for mdir in sorted(os.listdir(MODEL_DIR)):
        model_path = os.path.join(MODEL_DIR, mdir)
        model_name = mdir.split('_')[0]
        model_weights = mdir.split('_')[1].split('.')[0]
        dataset = mdir.split('_', 1)[1][:-4] if "synth" in mdir else "baseline"
    
        train_tf, val_tf = get_transforms()
        test_dataset = ImageFolderWithPaths(os.path.join(test_dir, 'test'), transform=val_tf)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
        num_classes = len(test_loader.dataset.classes)
        print(f"num classes: {num_classes}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(model_name, num_classes, weights=model_weights)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"Error loading model weights from {model_path}: {e}")
            exit(1)

        mispred_file_path = os.path.join(outdir, f"{model_name}_{model_weights}_{dataset}.json")
        print(f"Saving mispredictions to {mispred_file_path}")
        cm = evaluate_model(model, test_loader, device, mispred_file_path)
        print(f"Evaluation complete. Mispredictions saved to {mispred_file_path}")
