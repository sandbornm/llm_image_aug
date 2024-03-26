from typing import List, Tuple
import subprocess
import os
import glob
import json
import time


def parse_mispreds(mispreds_file: str) -> List[Tuple[str, str]]:
    mispreds = []
    with open(mispreds_file, 'r') as file:
        for line in file:
            img_path, pred = line.strip().split(',')[:2]
            img_path = img_path.split(': ')[1]
            label = img_path.split('/')[-2]
            pred = pred.split(': ')[1]
            mispreds.append((img_path, pred, label))
    return mispreds

def parse_mispreds_json(mispreds_file: str) -> List[Tuple[str, str]]:
    with open(mispreds_file, 'r') as file:
        mispred_data = json.load(file)
    mispred_dict = mispred_data['mispredictions']
    mispreds = []
    for img_path, pred_result in mispred_dict.items():
        pred = pred_result['Predicted']
        label = pred_result['Actual']
        mispreds.append((img_path, pred, label))
    return mispreds
    
def get_img_id_from_path(img_path: str) -> str:
    return img_path.split('/')[-1].split('.')[0]

def call_llava_cli(image_path, input_prompt, output_file):
    command = [
        "python", "llava_cli.py",
        "--image_path", image_path,
        "--input_prompt", input_prompt,
        "--output_file", output_file
    ]
    
    print(' '.join(command))

    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if process.stderr:
        print("Error:", process.stderr)
    else:
        print("Output written to:", output_file)

def llava_output_to_json(llava_output_dir: str):
    for class_name in os.listdir(llava_output_dir):
        class_dir = os.path.join(llava_output_dir, class_name)
        for file in os.listdir(class_dir):
            if file.endswith('.txt'):
                prompt_type = file.split('-')[-1].split('.')[0]
                file_data = {'prompt_type': prompt_type, 'prompt': '', 'response': ''}
                with open(os.path.join(class_dir, file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # Capture the prompt
                        if line.startswith("USER: ") and not line.startswith("USER: ASSISTANT: ") and line.strip() != "USER: exit...":
                            prompt = line.strip().split("USER: ", 1)[1]
                            file_data['prompt'] = prompt
                        # Capture the response
                        elif line.startswith("USER: ASSISTANT: "):
                            response = line.strip().split("USER: ASSISTANT: ", 1)[1]
                            file_data['response'] = response
                
                # Skip saving if the prompt or response is missing
                if not file_data['prompt'] or not file_data['response']:
                    continue

                # Write the file data to a JSON file
                json_file_name = file.replace('.txt', '.json')
                if not os.path.exists(os.path.join(class_dir, json_file_name)):
                    with open(os.path.join(class_dir, json_file_name), 'w') as f_json:
                        json.dump(file_data, f_json, indent=4)
                
    
if __name__ == "__main__":

    DESCRIBE_PROMPT = "Analyze the image in a comprehensive and detailed manner."
    DIAGNOSE_PROMPT = "Explain with specific visual details why this image of a {TRUE_CLASS} could be mistaken for a {PRED_CLASS}."
    MISPREDICTIONS_ROOT = "/home/michael/Github/llm_image_aug/data/test_performance"
    LLAVA_RESULTS_ROOT = "/home/michael/Github/llm_image_aug/data/llava_results"
    ALL_MISPRED_FILES = glob.glob(os.path.join(MISPREDICTIONS_ROOT, "resnet34*.json"))

    ts = time.time()
    for mispred_file in sorted(ALL_MISPRED_FILES):
        print(f"processing {mispred_file}...")
        model_name, model_weights, data_setting = mispred_file.split('/')[-1].split('.')[0].split('_')
        mispred_list = parse_mispreds_json(mispred_file)
        llava_result_dir = os.path.join(LLAVA_RESULTS_ROOT, f"{model_name}-{model_weights}-{data_setting}")
        print(f"Number of mispredictions: {len(mispred_list)}")
        print(f"LLAVA results dir: {llava_result_dir}")
        ss = time.time()
        for i, mispred in enumerate(mispred_list):
            print(f"current mispred: {i+1}/{len(mispred_list)}")
            image_path, pred, label = mispred
            print(f"image_path: {image_path}")
            img_id = get_img_id_from_path(image_path)
            for prompt in [("desc", DESCRIBE_PROMPT),
                        ("diag", DIAGNOSE_PROMPT)]:
                prompt_type, prompt_text = prompt
                outdir = os.path.join(llava_result_dir, label)
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(outdir, f"{img_id}-{prompt_type}.txt")
                if not os.path.exists(outfile):
                    if prompt_type == "diag":
                        prompt_text = prompt_text.format(TRUE_CLASS=label.replace("_", " "), PRED_CLASS=pred.replace("_", " "))
                    call_llava_cli(image_path, prompt_text, outfile)
                else: 
                    print(f"Skipping file already exists: {outfile}")
        se = time.time()
        print(f"Time elapsed for {len(mispred_list)} mispredictions: {se-ss}")
    te = time.time()
    print(f"Total time elapsed: {te-ts}")
