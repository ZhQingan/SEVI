import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import sys
import os
from transformers import set_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

from lavis.models import load_model_and_preprocess
evolve_vcd_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = "instructblip-7b"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads InstructBLIP model
    # For large_sized model,
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")
    
    for line in tqdm(questions):
        image_file = line["image"]
        qs = line["question"]
        prompt = qs

        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # prepare the image
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        if args.use_cd:
            another_image_file = line["another_image"]
            another_pt = os.path.join(args.image_folder, another_image_file)
            if not os.path.exists(another_pt):
                print(f"There is no image: {another_pt}")
                exit()
            image_another = Image.open(another_pt).convert("RGB")
            image_tensor_cd = vis_processors["eval"](image_another).unsqueeze(0).to(device)
        else:
            image_tensor_cd = None

        with torch.inference_mode():
            outputs = model.generate(
                {"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, 
                num_beams=1,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                max_new_tokens=args.max_gen_len,
                repetition_penalty=1,
                images_cd=image_tensor_cd,
                cd_beta = args.cd_beta,
                cd_alpha = args.cd_alpha,
                add_layer=[eval(_) for _ in args.add_layer.split()],
                cross_jsd_th=args.cross_jsd_th,
                kappa=args.kappa,
                omega=args.omega,
                )
        outputs = outputs[0]
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
    # parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
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
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--kappa", type=float, default=0.2)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("--cross_jsd_th", type=float, default=0.0001)
    parser.add_argument("--add_layer", type=str, default="-1")
    parser.add_argument("--start_layer", type=int, default=999)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    layers = [str(_) for _ in range(args.start_layer, 128)]
    args.add_layer = ' '.join(layers)
    if args.use_cd:
        from vcd_utils.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()

    print(str(args), flush=True)
    eval_model(args)
