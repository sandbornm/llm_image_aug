from openai import OpenAI
import requests
import backoff
import os
import uuid

client = OpenAI()


@backoff.on_exception(backoff.expo,
                      requests.exceptions.RequestException,
                      max_tries=3)
def get_dalle_image(prompt: str, img_save_path: str, num_images: int=1) -> str:
    response = client.images.generate(
        model="dall-e-2",
        prompt=prompt,
        size="256x256",
        quality="standard",
        n=num_images,
    )
        
    img_uuid = str(uuid.uuid4())

    img_url = response.data[0].url
    r = requests.get(img_url)

    img_file_path = os.path.join(img_save_path, f"{img_save_path.split('/')[-1]}_{img_uuid}.png")
    try:
        if r.status_code == 200:
            if not os.path.exists(img_file_path):
                with open(img_file_path, 'wb') as file:
                    file.write(r.content)
                print(f"Image saved to: {img_file_path}")
            else:
                print(f"Image already exists: {img_file_path}")
        else:
            print(f"Failed to download image with status code {r.status_code}: {img_url}")
    except Exception as e:
        print(f"An error occurred while saving the image to {img_file_path}: {e}")

                
                
if __name__ == "__main__":
    train_path = "/home/michael/Github/llava_test/models/imagenet_100/train"
    class_names = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
    print(f"{len(class_names)=}")
    
    prompt_clarification_class_mapping = {
        'boxer': 'boxer dog',
        'carbonara': 'carbonara pasta',
        'hard disc': 'hard disc drive',
        'iron': 'clothes iron',
        'mortarboard': 'mortarboard hat',
        'pickup': 'pickup truck',
        'reel': 'fishing reel',
        'tabby': 'tabby cat',
        'walking stick': 'walking stick insect',
        'wing': 'airplane wing',
    }
    
    IMAGES_PER_CLASS = 1
    
    for class_name in class_names[:1]:
        class_img_dir = os.path.join("/home/michael/Github/llava_test/models/imagenet_100_synthetic", class_name)
        if not os.path.exists(class_img_dir):
            os.makedirs(class_img_dir, exist_ok=True)
            
        if class_name in prompt_clarification_class_mapping:
            class_clarified = prompt_clarification_class_mapping[class_name]
            prompt = f"An image of a {class_clarified}"
        else:
            prompt = f"An image of a {class_name}"
        print(f"{prompt=}")
        
        for x in range(IMAGES_PER_CLASS):
            get_dalle_image(prompt, class_img_dir)