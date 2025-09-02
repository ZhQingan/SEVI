### Under Construction ...
---

# Aligning Attention Distribution to Information Flow for Hallucination Mitigation in Large Vision-Language Models

#### [Paper link](https://arxiv.org/abs/2505.14257)

---

## Setup

#### Environment

```bash
conda create -n sevi -y python=3.10
conda activate sevi

pip install -r requirements.txt
```

**Note**: The dependencies are refered to LLaVA, InstructBLIP, Qwen2-VL-Instruct, Qwen2.5-Instruct. You could also easily setup the environment by following the instructions from official repositories.

#### Datasets

All benchmarks need to be processed into structurally consistent JSON files.

Some samples could be found in `data/samples.json`.

You can use `preprocess.py` to build data:
```bash
python preprocess.py \
--image_file /path/to/images \
--data_file /path/to/data.json \
--output_file /path \
--retrieve False
```

## Implementation


#### Quick start

We developed a shell script `scripts/eval_all.sh` that can execute all evaluation tasks end-to-end.

**Inference**

You can also test your own dataset.
```bash
model_name=model_name
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path /path/to/model \
--question-file /path/to/test.json \
--image-folder /path \
--answers-file /path/to/results.json \
--omega 0.5 \
--start_layer 8 \
--kappa 0.2

```

**Evaluation**

```bash
# CHAIR
python test_chair.py --cap_file ../path/to/results.json --coco_path /path/to/coco
```
AMBER and DetailCaps has its own evaluation method; please refer the official repositories for result evaluation:  [AMBER](https://github.com/junyangwang0410/AMBER), [DetailCaps](https://github.com/foundation-multimodal-models/CAPTURE?tab=readme-ov-file)
