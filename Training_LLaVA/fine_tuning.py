from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from transformers import TextStreamer

def generate_response(self, image, input_text):
    if 'llama-2' in self.llava_model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in self.llava_model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in self.llava_model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles if "mpt" not in self.llava_model_name.lower() else ('user', 'assistant')

    image_tensor = self.llava_image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    inp = input_text
    if image is not None:
        if self.llava_model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        conv.append_message(conv.roles[0], inp)

    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        output_ids = self.llava_model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=2048,
            streamer=streamer,
            use_cache=True,
        )

    outputs = self.tokenizer.decode(output_ids[0]).strip()
    conv.messages[-1][-1] = outputs
    outputs = outputs.replace("\n\n", " ")
    return outputs

def collect_data():
    


if __name__ == "__main__":
