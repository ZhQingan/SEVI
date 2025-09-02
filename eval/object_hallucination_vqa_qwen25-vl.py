import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import math

from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM,AutoProcessor
from qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


tpn_map={
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf16': torch.bfloat16,
}
def eval_model(args):
    # Model
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    answers_file = os.path.expanduser(args.answers_file)
    if os.path.exists(answers_file):
        print(f"File Exisits: {answers_file}")
        exit()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'qwen25-vl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, attn_implementation='eager', torch_dtype=tpn_map[args.torch_type], device_map="cuda",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path, min_pixels=14*14*1280, max_pixels=28*28*1280)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        image_file = line["image"]
        question = line["question"]

        image_path = os.path.join(args.image_folder, image_file)
        messages_batch = []
        messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "file://"+image_path},
                    {"type": "text",  "text": question},
                ],
            },
        ]
        messages_batch.append(messages)

        if args.use_cd:
            image_file = line["another_image"]
            another_pt = os.path.join(args.image_folder, image_file)
            if not os.path.exists(another_pt):
                print(f"There is no image: {another_pt}")
                exit()
            messages_cd = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "file://"+another_pt},
                        {"type": "text",  "text": question},
                    ],}]
            messages_batch.append(messages_cd)

        text = processor.apply_chat_template(
            messages_batch, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            do_sample=args.do_sample,
            max_new_tokens=args.max_gen_len,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=False,
            use_cache=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            cd_beta = args.cd_beta,
            cd_alpha = args.cd_alpha,
            add_layer = [eval(_) for _ in args.add_layer.split()],
            cross_jsd_th=args.cross_jsd_th,
            kappa=args.kappa,
            omega=args.omega,
        )
        model.img_len=-1
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs = outputs[0].strip()
        ans_file.write(json.dumps({"question_id": line['question_id'],
                                   "question": line['question'],
                                   "output": outputs,
                                   "label": line['label'],
                                   "prompt": text[0].replace('<|image_pad|>', ''),
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
    parser.add_argument("--cross_jsd_th", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_type", type=str, default="bf16", choices=['fp16','fp32','bf16'])
    parser.add_argument("--add_layer", type=str, default="-1")
    parser.add_argument("--start_layer", type=int, default=999)
    
    args = parser.parse_args()
    layers = [str(_) for _ in range(args.start_layer, 99)]
    args.add_layer = ' '.join(layers)
    set_seed(args.seed)
    if args.use_cd:
        from vcd_utils.vcd_sample_qwen2 import evolve_vcd_sampling
        evolve_vcd_sampling()
    
    print(str(args), flush=True)
    eval_model(args)
