#!/bin/bash

prompt_folder='../FactualityPrompt/prompts/'
model_name='EleutherAI/pythia-6.9b-deduped'
#model_name='openlm-research/open_llama_7b_v2'
#model_name='facebook/opt-6.7b'

#export CUDA_LAUNCH_BLOCKING=1

dataset_suffix='_test7k'

temperature='1'

METHOD_ARR=( 'fe_topp' 'topp' )
MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
SUBMETHOD_ARR=( 'exp_1_win' 'a' )
P_ARR=( '1.0' '0;0.3;0.5;0.6;0.7;1' )
DT_ARR=( '0.5;0.7;1.0;2.0;3.0;4.0' '1.0' )

#METHOD_ARR=( 'topp' 'eta' 'typical' 'decay_period' 'topk'  )
#MODEL_ARR=( '' '' '' '' '' )
#SUBMETHOD_ARR=( 'a' 'a' 'a' 'a' 'a' )
#P_ARR=( '1.0;0.8;0.7;0.6;0.5;0.4;0.3' '0.1;0.3;0.8' '2;0.9;0.5;0.3' '0.9' '10;5;3;2;1'  )
#DT_ARR=( '1' '1' '1' '0.95;0.9;0.7;0.5;0.3;0.1' '1' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'EleutherAI/pythia-70m-deduped' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.6;0.4;0.2;0.25;0.1;0.05' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'fe_topp_period' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '')
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.9' '1.0' )
#DT_ARR=( '4.0;0.7;1.5' '5.0' '0.1;0.3;0.7;0.9' )

#METHOD_ARR=( 'CD' )
#MODEL_ARR=( 'facebook/opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.1;0.05' )
#DT_ARR=( '0.25' )

#METHOD_ARR=( 'CS' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '5.0' )
#DT_ARR=( '0.6' )
#P_ARR=( '5.0;10' )
#DT_ARR=( '0.4;0.6' )

init_existing_seeds=0
repeat_times=4

dataset_names=("fever_factual${dataset_suffix}_final.jsonl" "fever_nonfactual${dataset_suffix}_final.jsonl")

#input_datasets=($(for x in "${dataset_names[@]}"; do printf "$x%.0s " {1..${repeat_times}}; done))
#for v in ${dataset_names[@]}; do for i in $(seq 1 $repeat_times); do echo $v; done; done

END=$(($init_existing_seeds + $repeat_times - 1))
input_datasets=($(for v in ${dataset_names[@]}; do for i in $(seq 1 $repeat_times); do echo $v; done; done))
existing_seeds_arr=($(seq $init_existing_seeds $END))
existing_seeds_arr=("${existing_seeds_arr[@]}" "${existing_seeds_arr[@]}")
echo ${input_datasets[@]}
echo ${existing_seeds_arr[@]}

for j in "${!METHOD_ARR[@]}"; do
	MODEL=${MODEL_ARR[$j]}
	sample_method=${METHOD_ARR[$j]}
	sample_sub_method=${SUBMETHOD_ARR[$j]}
	top_p_all=${P_ARR[$j]}
	decay_temperature_all=${DT_ARR[$j]}
	
	final_entropy_model_path="models/$MODEL"
	batch_size=8
	if [[ $MODEL == *"410"* ]]; then
		batch_size=4
	fi
	if [[ $MODEL == *"_1b_"* ]]; then
		batch_size=2
	fi
	if [[ $MODEL == *"EleutherAI"* ]]; then
		batch_size=4
	fi
	if [[ $sample_method == *"fe_CD"* ]]; then
		batch_size=4
	fi
	if [[ $sample_method == *"CS"* ]]; then
		batch_size=1
	fi
	IFS=";" read -r -a top_p_list <<< "${top_p_all}"
	IFS=";" read -r -a decay_temperature_list <<< "${decay_temperature_all}"
	for top_p in "${top_p_list[@]}"; do
		for decay_temperature in "${decay_temperature_list[@]}"; do
			pids=()
			for i in "${!input_datasets[@]}";
			do
				dataset_name=${input_datasets[$i]}
				num_existing_seeds=${existing_seeds_arr[$i]}
				echo "python src/factual_gen/gen_fp.py --model_name=$model_name --input_file_name ${prompt_folder}/$dataset_name --cuda_idx $i --p $top_p --num_existing_seeds $num_existing_seeds --sample_method $sample_method --final_entropy_model_path $final_entropy_model_path --batch_size $batch_size --decay_temperature $decay_temperature --temperature $temperature --sample_sub_method $sample_sub_method"
				#sleep 1 &
				python src/factual_gen/gen_fp.py --model_name=$model_name --input_file_name ${prompt_folder}/$dataset_name --cuda_idx $i --p $top_p --num_existing_seeds $num_existing_seeds --sample_method $sample_method --final_entropy_model_path $final_entropy_model_path --batch_size $batch_size --decay_temperature $decay_temperature --temperature $temperature --sample_sub_method $sample_sub_method  &
				pids+=($!)
			done
			echo "${pids[@]}"
			wait "${pids[@]}"
		done
	done
done
