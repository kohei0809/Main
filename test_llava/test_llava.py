import torch
#from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import TextStreamer

model_path = "liuhaotian/llava-v1.5-13b"
load_4bit = True
load_8bit = not load_4bit
disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit)


def load_image(image_file):
    if image_file.startswith('http'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_response(image, input_text, model_path):
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles if "mpt" not in model_name.lower() else ('user', 'assistant')

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    inp = input_text
    print(f"ROLE: {roles[1]}: ", end="")

    if image is not None:
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2048,
            streamer=streamer,
            use_cache=True,
            #stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs
    outputs = outputs.replace("\n\n", " ")

    return outputs

def human_test():
    user_list = ["matsumoto", "kondo", "nakamura", "aizawa", "edward"]
    image_folder = "/gs/fs/tga-aklab/matsumoto/Main/result_images"
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    
    for user_name in user_list:
        for i in range(11):
            image_file = f"{image_folder}/{user_name}/result_{i}.png"
            print(image_file)
            input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
            image = load_image(image_file)
            response = generate_response(image, input_text, model_path)     
            print(f"A:{response[4:-4]}")
            print("------------------------------")   

if __name__ == '__main__':
    
    human_test()
    """
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    model_path = "liuhaotian/llava-v1.5-13b"
    image_file = "/gs/fs/tga-aklab/matsumoto/Main/pictures/result_image.png"
    input_text = "This picture shows ten pictures in one building, five horizontally and two vertically side by side. Black lines are drawn between the pictures. From these pictures, describe the environment of this building in detail."
    input_text = "You are an excellent property writer. This picture consists of 10 pictures arranged in one picture, 5 horizontally and 2 vertically on one building. In addition, a black line separates the pictures from each other. From each picture, you should understand the details of this building's environment and describe this building's environment in detail in the form of a summary of these pictures. At this point, do not describe each picture one at a time, but rather in a summarized form. Also note that each picture was taken in a separate location, so successive pictures are not positionally close. Additionally, do not mention which picture you are quoting from or the black line separating each picture."
    image = load_image(image_file)
    response = generate_response(image, input_text)

    #plt.imshow(image)
    #plt.axis('off') 
    #plt.show()

    print(f"Q:{input_text}")
    print(f"A:{response[4:-4]}")
    """
    print("FINISH !!")
