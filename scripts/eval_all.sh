{
export CUDA_VISIBLE_DEVICES=$1

seed=17
cd_beta=0.1
max_len=512
use_cd='true'
cross_jsd_th=0.0001
start_layer=8
omega=0.5
kappa=0.2
model_name=$2


if [[ $model_name == "llava" ]]; then
  model_path="/path/to/models/llava-v1.5-7b"
elif [[ $model_name == "blip" ]]; then
  model_path="/path/to/models/blip"
elif [[ $model_name == "qwen-chat" ]]; then
  model_path="/path/to/models/Qwen-VL-Chat"
elif [[ $model_name == "qwen2-vl" ]]; then
  model_path="/path/to/models/Qwen2-VL-7B-Instruct"
elif [[ $model_name == "qwen25-vl" ]]; then
  model_path="/path/to/models/Qwen2.5-VL-7B-Instruct"
elif [[ $model_name == "llava-next" ]]  || [[ $model_name == 'llava-next-15' ]]; then
  model_path="/path/to/models/llama3-llava-next-8b"
else
  model_path=""
fi

image_folder=/path/to/benchmark/AMBER/data/image
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path $model_path \
--question-file ../data/AMBER/amber.json \
--image-folder $image_folder \
--answers-file ./path/to/results/amber_new/${model_name}.amber.len_${max_len}.seed${seed}.jsonl \
--use_cd $use_cd \
--cd_beta $cd_beta \
--max_gen_len $max_len \
--cross_jsd_th $cross_jsd_th \
--start_layer $start_layer \
--omega $omega \
--kappa $kappa \
--seed $seed


max_len=512
dataset_name="DC"
image_folder=/path/to/benchmark/pope/val2014_aug
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path $model_path \
--question-file ../data/${dataset}.json \
--image-folder $image_folder \
--answers-file ./path/to/results/${model_name}.${dataset}.len_${max_len}.seed${seed}.jsonl \
--use_cd $use_cd \
--cd_beta $cd_beta \
--max_gen_len $max_len \
--cross_jsd_th $cross_jsd_th \
--start_layer $start_layer \
--omega $omega \
--kappa $kappa \
--seed $seed


dataset_name="chair"
image_folder=/path/to/benchmark/pope/val2014_aug
for max_len in 64 512; do
python ./eval/object_hallucination_vqa_${model_name}.py \
--model-path $model_path \
--question-file ../data/${dataset}.json \
--image-folder $image_folder \
--answers-file ./path/to/results/${model_name}.${dataset}.len_${max_len}.seed${seed}.jsonl \
--use_cd $use_cd \
--cd_beta $cd_beta \
--max_gen_len $max_len \
--cross_jsd_th $cross_jsd_th \
--start_layer $start_layer \
--omega $omega \
--kappa $kappa \
--seed $seed
done

}