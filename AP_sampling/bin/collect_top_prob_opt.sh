#!/bin/bash
bptt=1024

#data_folder_name="ROC_gen_1000_p095_OPT"
#data_folder_name="news_gen_1000_p095_OPT"
#data_folder_name="wp_gen_1000_p095_OPT"
#data_folder_name="openwebtext_2017_18_1e5_OPT"
#data_folder_name="wiki2021_1e6_OPT"
data_folder_name="wiki2021_5e6_OPT"
#data_folder_name="ROC_spring_OPT"
#data_folder_name="wikinews_OPT"
#data_folder_name="wp_5000_OPT"
#data_folder_name="wp_20000_OPT"
#data_folder_name="wiki2021_1e5_OPT"

#top_k="10"
#sampling_methods="10_20"
top_k="20,5,10"
sampling_methods="0_20,20_100,100_inf"

#top_w_idx_model_name="EleutherAI/pythia-6.9b-deduped"
top_w_idx_model_name="facebook/opt-6.7b"
#output_folder="data/processed/$data_folder_name/prob_opt_tensor_$bptt"
output_folder="data/processed/$data_folder_name/prob_opt_tensor_${bptt}_new"
#input_folder_name="../true_entropy/data/processed/$data_folder_name"
input_folder_name="data/processed/$data_folder_name"

declare -a bsz_arr=(2 4 8 16)
declare -a model_arr=("facebook/opt-2.7b" "facebook/opt-1.3b" "facebook/opt-350m" "facebook/opt-125m" )

model_name="facebook/opt-6.7b"
batch_size=1
cuda_init=0
echo "python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $input_folder_name --output_folder $output_folder --cuda_idx $cuda_init --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt"
python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $input_folder_name --output_folder $output_folder --cuda_idx $cuda_init --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt

pids=()

for i in "${!model_arr[@]}";
do
	model_name=${model_arr[$i]}
	batch_size=${bsz_arr[$i]}
	echo "python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $input_folder_name --output_folder $output_folder --cuda_idx $i --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt"
	python src/collect_top_prob.py --model_name=$model_name --top_w_idx_model_name=$top_w_idx_model_name --input_folder_name $input_folder_name --output_folder $output_folder --cuda_idx $i --batch_size $batch_size --top_k $top_k --sampling_methods $sampling_methods --bptt $bptt &
	pids+=($!)
done
echo "${pids[@]}"

