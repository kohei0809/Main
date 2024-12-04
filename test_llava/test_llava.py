import os
import torch
import numpy as np
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

import torchvision.transforms as transforms
from habitat.core.logging import logger

import cv2

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

    logger.info(f"image_tensor={image.size}")
    image_array = np.array(image)
    image_array = cv2.resize(
            image_array,
            (336, 336),
            interpolation=cv2.INTER_CUBIC,
        )
    image = Image.fromarray(image_array)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    logger.info(f"image_tensor={image_tensor[0].shape}")
    image_tensor2 = image_tensor[0] * 255
    image_tensor2 = image_tensor2.to(torch.uint8)
    image_tensor2 = transforms.ToPILImage()(image_tensor2)
    # 画像を保存
    image_tensor2.save("output_image2.png")

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
    user_list = ["nakamura", "aizawa", "edward"]
    image_folder = "/gs/fs/tga-aklab/matsumoto/Main/result_images"
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    
    for user_name in user_list:
        for i in range(11):
            image_dir = f"{image_folder}/{user_name}/scene_{i}/"
            image_result_file = image_dir + f"result_{i}.png"
            print(image_result_file)
            result_image = load_image(image_result_file)

            input_text1 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

            image_descriptions = []
            for j in range(10):
                image_file = image_dir + f"{j}.png"
                print(image_file)
                if os.path.isfile(image_file) == False:
                    print("No File")
                    break
            
                image = load_image(image_file)
                response = generate_response(image, input_text1, model_path)
                response = response[4:-4]
                image_descriptions.append(response)

            input_text2 = "# Instructions\n"\
                        "You are an excellent property writer.\n"\
                        "# Each_Description is a description of the building in the pictures you have entered. Please summarize these and write a description of the entire environment as if you were a person in this environment.\n"\
                        "\n"\
                        "# Each_Description\n"
            input_text3 = "# Notes\n"\
                        "・Please summarize # Each_Description and write a description of the entire environment as if you were a person in this environment.\n"\
                        "・Please write approximately 100 words.\n"\
                        "・Please note that the sentences in # Each_Description are not necessarily close in distance."

            image_num = len(image_descriptions)
            for j in range(10):
                desc_num = j % image_num
                each_description = "・" + image_descriptions[desc_num] + "\n"
                input_text2 += each_description

            input_text = input_text2 + "\n" + input_text3

            response = generate_response(result_image, input_text, model_path)     
            print(f"A:{response[4:-4]}")
            print("------------------------------")   

if __name__ == '__main__':
    
    #human_test()
    
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    model_path = "liuhaotian/llava-v1.5-13b"
    image_file = "/gs/fs/tga-aklab/matsumoto/Main/map_image.png"

    input_text = "The image is a map of a certain environment.\n"\
                "White areas are already explored, and black areas are unexplored or walls.\n"\
                "This map is colored in “Red”, “Green”, “Blue”, “Yellow”, “Cyan”, “Magenta”, “Orange”, “Purple”, “Brown”, and “Pink” one area at a time.\n"\
                "Please describe the absolute positions of the colored areas in detail."
    """
    input_text = "The image is a map of a certain environment. \n"\
                "The white areas have already been explored, and the black areas are unexplored or walls. \n"\
                "This map is also [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0 , 255], [255, 165, 0], [128, 0, 128], [139, 69, 19], [255, 192, 203] are each divided into one area by one color. \n"\
                "Please explain in detail the absolute positional relationship of these color-coded areas."
    """
    input_text = "You are a robot that moves around based on a map. \n"\
                "In the map, white areas are already explored, and black areas are unexplored or walls. \n"\
                "Please explain in detail how you moved when moving in the order Red → Green → Blue → Yellow → Cyan → Magenta → Orange → Purple → Brown → Pink on this map."
    input_text = "You are a robot that moves around based on a map. \n"\
                "In the map, white areas are already explored, and black areas are unexplored or walls. \n"\
                "Using this map, when you move in the order “Red”, “Green”, “Blue”, “Yellow”, “Cyan”, “Magenta”, “Orange”, “Purple”, “Brown”, and “Pink”, please explain in detail how you moved around in this environment, based on spatial information."
    image = load_image(image_file)
    response = generate_response(image, input_text, model_path)

    #plt.imshow(image)
    #plt.axis('off') 
    #plt.show()

    print(f"Q:{input_text}")
    print(f"A:{response[4:-4]}")
    print("FINISH !!")
