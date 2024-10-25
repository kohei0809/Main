import os
import torch
from transformers import AutoProcessor, LlavaNextForConditionalGeneration 
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import TextStreamer

from habitat.core.logging import logger

#model_path = "llava-hf/llava-v1.6-mistral-7b-hf"
model_path = "llava-hf/llava-v1.6-vicuna-13b-hf"

# Load the model in half-precision
model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)


def load_image(image_file):
    if image_file.startswith('http'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def extract_after_assistant(S: str) -> str:
    # '[/INST]' が見つかった場所を特定する
    inst_index = S.find('ASSISTANT: ')
    
    # '[/INST]' が見つかった場合、その後の文章を返す
    if inst_index != -1:
        return S[inst_index + len('ASSISTANT: '):]
    
    # 見つからなかった場合は空の文字列を返す
    return ""


def generate_multi_response(image_list, input_text, max_new_tokens=4096):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": input_text},
                ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    prompts = [prompt for _ in range(len(image_list))]
    
    inputs = processor(images=image_list, text=prompts, padding=True, return_tensors="pt").to(model.device)

    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    logger.info(f"outputs = {outputs}")
    #logger.info(f"outputs_size = {len(outputs)}")
    #logger.info(f"image_list_size = {len(image_list)}")

    #logger.info(f"all: {outputs}")

    results = []
    for i in range(len(outputs)):
        output = extract_after_assistant(outputs[i].strip().replace("\n\n", " "))
        logger.info(f"Sentence {i}: ")
        logger.info(output)
        results.append(output)

    return results 

def human_test():
    user_list = ["matsumoto", "kondo", "nakamura", "aizawa", "edward"]
    #user_list = ["edward"]
    image_folder = "/gs/fs/tga-aklab/matsumoto/Main/result_images"
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    
    for user_name in user_list:
        for i in range(11):
            image_dir = f"{image_folder}/{user_name}/scene_{i}/"
            image_result_file = image_dir + f"result_{i}.png"
            logger.info(image_result_file)
            result_image = load_image(image_result_file)

            input_text1 = "# Instructions\n"\
                    "You are an excellent property writer.\n"\
                    "Please understand the details of the environment of this building from the pictures you have been given and explain what it is like to be in this environment as a person in this environment."

            image_list = []
            for j in range(10):
                image_file = image_dir + f"{j}.png"
                logger.info(image_file)
                if os.path.isfile(image_file) == False:
                    logger.info("No File")
                    break
            
                image = load_image(image_file)
                image_list.append(image)

            response_list = generate_multi_response(image_list, input_text1)
            
            input_text2 = "# Instructions\n"\
                        "You are an excellent property writer.\n"\
                        "# Each_Description is a description of the building in the pictures you have entered. Please summarize these and write a description of the entire environment as if you were a person in this environment.\n"\
                        "\n"\
                        "# Each_Description\n"
            input_text3 = "# Notes\n"\
                        "・Please summarize # Each_Description and write a description of the entire environment as if you were a person in this environment.\n"\
                        "・Please write approximately 100 words.\n"\
                        "・Please note that the sentences in # Each_Description are not necessarily close in distance."

            image_num = len(response_list)
            for j in range(10):
                desc_num = j % image_num
                each_description = "・" + response_list[desc_num] + "\n"
                input_text2 += each_description

            input_text = input_text2 + "\n" + input_text3

            response = generate_multi_response([result_image], input_text, max_new_tokens=20480)     
            logger.info(f"A:{response}")
            logger.info("------------------------------")

def test():
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

    response = generate_multi_response(image_list, input_text)


    logger.info(f"Q:{input_text}")
    print(f"A:{response[4:-4]}")
    logger.info(f"A:{response}")

if __name__ == '__main__':
    
    human_test()
    #test()

    logger.info("FINISH !!")
