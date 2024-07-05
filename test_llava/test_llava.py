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
    
    #human_test()
    
    #model_path = "/gs/fs/tga-aklab/matsumoto/Main/model/llava-v1.5-7b/"
    model_path = "liuhaotian/llava-v1.5-13b"
    image_file = "/gs/fs/tga-aklab/matsumoto/Main/result_images/matsumoto/result_1.png"
    input_text = "<Instructions>\n"\
                "You are an excellent property writer.\n"\
                "The picture you have entered consists of 10 pictures of a building, 5 horizontally and 2 vertically placed in a single picture.\n"\
                "Each picture is also separated by a black line.\n"\
                "From each picture, understand the details of this building's environment, and in the form of a summary of these pictures, describe this building's environment in detail, paying attention to the <Notes>.\n"\
                "In doing so, please also consider the location of each picture as indicated by <Location Information>.\n"\
                "\n\n"\
                "<Location Information>\n"\
                "The top leftmost picture is picture_1, and from its right to left are picture_2, picture_3, picture_4, and picture_5.\n"\
                "Similarly, the bottom-left corner is picture_6, and from its right, picture_7, picture_8, picture_9, and picture_10.\n"\
                "The following is the location information for each picture.\n\n"\
                "picture_1 : (-1.720, 10.050)\n"\
                "picture_2 : (-3.336, 4.277)\n"\
                "picture_3 : (-0.250, -4.250)\n"\
                "picture_4 : (6.473, -9.606)\n"\
                "picture_5 : (7.103, -7.400)\n"\
                "picture_6 : (2.094, -18.094)\n"\
                "picture_7 : (-5.632, -17.180)\n"\
                "picture_8 : (-4.080, 0.388)\n"\
                "picture_9 : (-1.059, -13.591)\n"\
                "picture_10 : (-0.066, -7.193)\n"\
                "<Notes>\n"\
                "・Note that each picture is taken at the location indicated by <Location Information>, and that adjacent pictures are not close in location.\n"\
                "・Please output the location information for each picture from <Location Information>.\n"\
                "・When describing the environment, do not mention whether it was taken from that picture or the black line separating each picture.\n"\
                "・Only refer to the structure of the description from <Example of output>, and do not use your imagination to describe things not shown in the picture.\n"\
                "\n\n"\
                "<Example of output>\n"\
                "This building features a spacious layout with multiple living rooms, bedrooms, and bathrooms. A living space with a fireplace is next to a fully equipped kitchen. There are also three bedrooms on the left side of the building, with a bathroom nearby. There are plenty of books to work with.\n"\
                "Overall, the apartment is spacious and well-equipped, with many paintings on the walls."
        
    image = load_image(image_file)
    response = generate_response(image, input_text, model_path)

    #plt.imshow(image)
    #plt.axis('off') 
    #plt.show()

    #print(f"Q:{input_text}")
    print(f"A:{response[4:-4]}")
    print("FINISH !!")
