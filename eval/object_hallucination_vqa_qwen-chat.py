import argparse
# import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
import math

from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen-vl-chat'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        image_file = line["image"]
        question = line["question"]

        image_path = os.path.join(args.image_folder, image_file)
        prompt = f'<img></img>{question} Answer:'
        input_ids = tokenizer([prompt], return_tensors='pt', padding='longest')
        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = model.transformer.visual.image_transform(image_tensor).unsqueeze(0).to(model.device)

        if args.use_cd:
            another_image_file = line["another_image"]
            another_pt = os.path.join(args.image_folder, another_image_file)
            if not os.path.exists(another_pt):
                print(f"There is no image: {another_pt}")
                exit()
            image_another = Image.open(another_pt).convert("RGB")
            image_tensor_cd = model.transformer.visual.image_transform(image_another).unsqueeze(0).to(model.device)

        else:
            image_tensor_cd = None
        
        pred = model.generate(
            input_ids=input_ids.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            do_sample=args.do_sample,
            max_new_tokens=args.max_gen_len,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            images = image_tensor,
            images_cd=image_tensor_cd,
            cd_beta = args.cd_beta,
            cd_alpha = args.cd_alpha,
            add_layer = [eval(_) for _ in args.add_layer.split()],
            cross_jsd_th=args.cross_jsd_th,
            kappa=args.kappa,
            omega=args.omega,
        )

        outputs = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in pred
        ][0]
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
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/ckpt/Qwen-VL")
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
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
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
