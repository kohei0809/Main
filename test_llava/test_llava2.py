import os
import torch
#from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import TextStreamer

from habitat.core.logging import logger

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


def generate_response(image_list, input_text, model_path):
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

    inp = input_text
    #print(f"ROLE: {roles[1]}: ", end="")

    if model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    image_sizes = [x.size for x in image_list]
    images_tensor = process_images(
        image_list,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2048,
            streamer=streamer,
            use_cache=True,
        )

    #outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    conv.messages[-1][-1] = outputs

    logger.info(f"outputs = {outputs}")
    logger.info(f"outputs_size = {len(outputs)}")
    logger.info(f"image_list_size = {len(image_list)}")

    output1 = outputs[0].strip()
    output1 = output1.replace("\n\n", " ")
    logger.info(f"AAAA: {output1}")

    output2 = outputs[1].strip()
    output2 = output2.replace("\n\n", " ")
    logger.info(f"BBBBB: {output2}")

    output3 = outputs[2].strip()
    output3 = output3.replace("\n\n", " ")
    logger.info(f"CCCC: {output3}")

    return outputs 

if __name__ == '__main__':
    
    #human_test()
    
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    model_path = "liuhaotian/llava-v1.5-13b"
    input_text = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

    image_list = []
    image_folder = "/gs/fs/tga-aklab/matsumoto/Main/result_images"
    user_name = "matsumoto"
    for i in range(10):
        image_dir = f"{image_folder}/{user_name}/scene_0/"
        image_result_file = image_dir + f"{i}.png"
        logger.info(image_result_file)
        result_image = load_image(image_result_file)
        image_list.append(result_image)

    response = generate_response(image_list, input_text, model_path)

    #plt.imshow(image)
    #plt.axis('off') 
    #plt.show()

    logger.info(f"Q:{input_text}")
    #print(f"A:{response[4:-4]}")
    logger.info(f"A:{response[4:-4]}")
    logger.info("FINISH !!")
