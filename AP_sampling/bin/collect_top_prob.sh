#!/bin/bash
#top_k=10
bptt=1024

#data_folder_name="wiki2021_1e4_Pythia"
#data_folder_name="ROC_gen_1000_p095_Pythia"
#data_folder_name="news_gen_1000_p095_Pythia"
#data_folder_name="wp_gen_1000_p095_Pythia"
#data_folder_name="wiki2021_1e6_Pythia"
data_folder_name="wiki2021_5e6_Pythia"
#data_folder_name="ROC_spring_Pythia"
#data_folder_name="wikinews_Pythia"
#data_folder_name="wp_5000_Pythia"
#data_folder_name="wp_20000_Pythia"
#data_folder_name="wiki2021_1e5_Pythia"

#top_k="10"
#sampling_methods="10_20"

top_k="20,5,10"
sampling_methods="0_20,20_100,100_inf"
#top_k="20,20,20"
#sampling_methods="0_20,20_100,100_inf"

top_w_idx_model_name="EleutherAI/pythia-6.9b-deduped"
output_folder="data/processed/$data_folder_name/prob_tensor_${bptt}_ext2"
#output_folder="data/processed/$data_folder_name/prob_tensor_${bptt}_ext3"
#input_folder_name="../true_entropy/data/processed/$data_folder_name"
input_folder_name="data/processed/$data_folder_name"

declare -a bsz_arr=(2 4 4 8 12 16)
declare -a model_arr=("EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1.4b-deduped" "EleutherAI/pythia-1b-deduped" "EleutherAI/pythia-410m-deduped" "EleutherAI/pythia-160m-deduped" "EleutherAI/pythia-70m-deduped" )

model_name="EleutherAI/pythia-6.9b-deduped"
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

