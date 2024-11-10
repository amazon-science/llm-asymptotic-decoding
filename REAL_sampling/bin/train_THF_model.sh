#!/bin/bash

#INPUT_FILE="data/raw/wiki2021_text_only_1e4"
#PROC_FOLDER="data/processed/wiki_1e4_Pythia_temp/"
#TOKENIZER="EleutherAI/pythia-70m-deduped"
#OUTPUT_MODEL_FOLDER="models/wiki_1e4_70M_bsz_128_exp_pred_last_a10_e3"

INPUT_FILE="data/raw/OWT_wiki_1e7"
PROC_FOLDER="data/processed/OWT_wiki_1e7_Pythia/"
TOKENIZER="EleutherAI/pythia-70m-deduped"
OUTPUT_MODEL_FOLDER="models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3"

echo "python src/prepare_id_corpus_from_raw.py --input_file $INPUT_FILE --output_dir $PROC_FOLDER/tensors_all/ --model_name $TOKENIZER"
python src/prepare_id_corpus_from_raw.py --input_file $INPUT_FILE --output_dir $PROC_FOLDER/tensors_all/ --model_name $TOKENIZER

declare -a bsz_arr=(1 2 4 4 8 12 16)
declare -a model_arr=("EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1.4b-deduped" "EleutherAI/pythia-1b-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-70m-deduped" )

pids=()
for i in "${!model_arr[@]}";
do
	model_name=${model_arr[$i]}
	batch_size=${bsz_arr[$i]}
	echo "python src/collect_gt_entropy.py --model_name=$model_name --input_folder_name $PROC_FOLDER --cuda_idx $i --batch_size $batch_size"
	python src/collect_gt_entropy.py --model_name=$model_name --input_folder_name $PROC_FOLDER --cuda_idx $i --batch_size $batch_size &
	pids+=($!)
done
echo "${pids[@]}"
wait "${pids[@]}"

echo "python src/train_entropy_prediction_model.py --output_dir $OUTPUT_MODEL_FOLDER --train_text_file $PROC_FOLDER/tensors_all/train.pt --validation_text_file $PROC_FOLDER/tensors_all/val_org.pt --train_label_folder $PROC_FOLDER/entropy_tensor_1024/train  --validation_label_folder $PROC_FOLDER/entropy_tensor_1024/val --model_name_or_path ${model_arr[-1]} --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --logging_steps 10 --warmup_steps 100  --eval_steps 500 --evaluation_strategy steps --save_steps 5000 --num_train_epochs 3"
python src/train_entropy_prediction_model.py --output_dir $OUTPUT_MODEL_FOLDER --train_text_file $PROC_FOLDER/tensors_all/train.pt --validation_text_file $PROC_FOLDER/tensors_all/val_org.pt --train_label_folder $PROC_FOLDER/entropy_tensor_1024/train  --validation_label_folder $PROC_FOLDER/entropy_tensor_1024/val --model_name_or_path ${model_arr[-1]} --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --logging_steps 10 --warmup_steps 100  --eval_steps 500 --evaluation_strategy steps --save_steps 5000 --num_train_epochs 3
