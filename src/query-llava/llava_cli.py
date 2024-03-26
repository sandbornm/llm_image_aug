import subprocess
import argparse


"""
python -m llava.serve.cli \
>     --model-path liuhaotian/llava-v1.5-7b \
>     --image-file "https://llava-vl.github.io/static/images/view.jpg" \
>     --load-4bit
"""

# Set up argument parser
parser = argparse.ArgumentParser(description="Run llava command with user input")
parser.add_argument("--image_path", type=str, required=True, help="URL of the image file")
parser.add_argument("--input_prompt", type=str, required=True, help="User input prompt")
parser.add_argument("--output_file", type=str, required=True, help="File to write output")

# Parse arguments
args = parser.parse_args()

# Define the command to run
command = [
    "python", "-m", "llava.serve.cli",
    "--model-path", "liuhaotian/llava-v1.5-7b",
    "--image-file", args.image_path,
    "--load-4bit"
]

# Start the subprocess
process = subprocess.Popen(command,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           text=True)

# Iterate over the input prompts and communicate with subprocess
try:
    stdout, stderr = process.communicate(input=args.input_prompt.strip() + "\n", timeout=60)  # 30 seconds timeout
except subprocess.TimeoutExpired:
    print("Process timed out")
    process.kill()
    stdout, stderr = process.communicate()

with open(args.output_file, 'w') as file:
    file.write("USER: " + args.input_prompt + "\n")
    file.write(stdout)


# Terminate the subprocess
process.terminate()
