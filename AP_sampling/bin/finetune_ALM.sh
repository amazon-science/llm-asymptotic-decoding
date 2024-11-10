#!/bin/bash

bptt=1024

#INPUT_FILE="data/raw/wiki2021_text_only_1e4"
#data_folder_name="wiki2021_1e4_Pythia"
#OUTPUT_MODEL_FOLDER="models/prob_wiki_ext2_1e4_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"
INPUT_FILE="data/raw/wiki2021_text_only_1e6"
data_folder_name="wiki2021_1e6_Pythia"
OUTPUT_MODEL_FOLDER="models/prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"

PROC_FOLDER="data/processed/$data_folder_name"
output_folder="data/processed/$data_folder_name/prob_tensor_${bptt}"
TOKENIZER="EleutherAI/pythia-70m-deduped"

top_k="20,5,5"
sampling_methods="0_20,20_100,100_inf"

top_w_idx_model_name="EleutherAI/pythia-6.9b-deduped"


declare -a bsz_arr=(2 4 4 8 12 16)
declare -a model_arr=("EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1.4b-deduped" "EleutherAI/pythia-1b-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-70m-deduped" )

model_name="EleutherAI/pythia-6.9b-deduped"
batch_size=1
cuda_init=0

echo "python ../REAL_sampling/src/prepare_id_corpus_from_raw.py --input_file $INPUT_FILE --output_dir $PROC_FOLDER/tensors_all/ --model_name $TOKENIZER"
python ../REAL_sampling/src/prepare_id_corpus_from_raw.py --input_file $INPUT_FILE --output_dir $PROC_FOLDER/tensors_all/ --model_name $TOKENIZER

echo "python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $PROC_FOLDER --output_folder $output_folder --cuda_idx $cuda_init --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt"
python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $PROC_FOLDER --output_folder $output_folder --cuda_idx $cuda_init --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt

pids=()

for i in "${!model_arr[@]}";
do
        model_name=${model_arr[$i]}
        batch_size=${bsz_arr[$i]}
        echo "python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $PROC_FOLDER --output_folder $output_folder --cuda_idx $i --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt"
        python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $PROC_FOLDER --output_folder $output_folder --cuda_idx $i --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt &
        pids+=($!)
done
echo "${pids[@]}"
wait "${pids[@]}"


echo "python src/train_logits_prediction_model.py --output_dir $OUTPUT_MODEL_FOLDER --train_text_file $PROC_FOLDER/tensors_all/train.pt --validation_text_file $PROC_FOLDER/tensors_all/val_org.pt --train_label_folder $output_folder/train --validation_label_folder $output_folder/val --model_name_or_path ${model_arr[-1]} --do_train --do_eval --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --logging_steps 10 --warmup_steps 100  --eval_steps 500 --evaluation_strategy steps --save_steps 500 --num_train_epochs 5 --learning_rate 1e-4 --logit_reg_w 0.8 --file_suffix _${sampling_methods}_k_${top_k}_bptt_${bptt}.pt"
python src/train_logits_prediction_model.py --output_dir $OUTPUT_MODEL_FOLDER --train_text_file $PROC_FOLDER/tensors_all/train.pt --validation_text_file $PROC_FOLDER/tensors_all/val_org.pt --train_label_folder $output_folder/train --validation_label_folder $output_folder/val --model_name_or_path ${model_arr[-1]} --do_train --do_eval --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --logging_steps 10 --warmup_steps 100  --eval_steps 500 --evaluation_strategy steps --save_steps 500 --num_train_epochs 5 --learning_rate 1e-4 --logit_reg_w 0.8 --file_suffix _${sampling_methods}_k_${top_k}_bptt_${bptt}.pt
