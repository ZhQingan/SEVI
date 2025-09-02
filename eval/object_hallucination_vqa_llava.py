import argparse
import torch
import os
import json
from tqdm import tqdm
import random

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

from transformers import set_seed

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        image_file = line["image"]
        qs = line["question"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            another_image_file = line["another_image"]
            another_pt = os.path.join(args.image_folder, another_image_file)
            if not os.path.exists(another_pt):
                print(f"There is no image: {another_pt}")
                exit()
            image_another = Image.open(another_pt)
            image_tensor_cd = image_processor.preprocess(image_another, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor_cd = None
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                add_layer=[eval(_) for _ in args.add_layer.split()],
                cross_jsd_th=args.cross_jsd_th,
                kappa=args.kappa,
                omega=args.omega,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_gen_len,
                use_cache=True,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        ans_file.write(json.dumps({"question_id": line['question_id'],
                                   "question": line['question'],
                                   "output": outputs,
                                   "label": line['label'],
                                   "prompt": prompt,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "image_id": line['image_id'],
                                   }) + "\n")
        ans_file.flush()
    ans_file.write(json.dumps(vars(args)) + '\n')
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--use_cd", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.2)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--cross_jsd_th", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add_layer", type=str, default="-1")
    parser.add_argument("--start_layer", type=int, default=999)
    

    args = parser.parse_args()
    layers = [str(_) for _ in range(args.start_layer, 128)]
    args.add_layer = ' '.join(layers)
    set_seed(args.seed)
    if args.use_cd:
        from vcd_utils.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()

    print(str(args), flush=True)
    eval_model(args)
