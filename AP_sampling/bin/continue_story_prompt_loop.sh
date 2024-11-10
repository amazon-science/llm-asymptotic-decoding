#!/bin/bash

test_dataset='story'

prompt_folder="/mnt/efs/Haw-Shiuan/AP_sampling/outputs/${test_dataset}/"
eval_folder='/mnt/efs/Haw-Shiuan/Hallucination_repos/FactualityPrompt'
output_folder="/mnt/efs/Haw-Shiuan/AP_sampling/outputs/${test_dataset}/"

#GEN_MODEL="6.9b"
#GEN_MODEL="OpenLLaMA2-7b"
GEN_MODEL="OPT-6.7b"

if [[ $GEN_MODEL == "6.9b" ]]; then
        model_name='EleutherAI/pythia-6.9b-deduped'
elif [[ $GEN_MODEL == "OpenLLaMA2-7b" ]]; then
        model_name='openlm-research/open_llama_7b_v2'
elif [[ $GEN_MODEL == "OPT-6.7b" ]]; then
        model_name='facebook/opt-6.7b'
fi

#export CUDA_LAUNCH_BLOCKING=1

dataset_suffix='_1000'
#dataset_suffix='_100'
#dataset_suffix='_10'

if [[ $test_dataset == "story" ]]; then
	PROMPT_TYPE="story_start2${dataset_suffix}"
	dataset_names=( "prompt_start2_b1${dataset_suffix}.jsonl" )
	max_len=128
	NUM_EVAL_SENT=3
fi
temperature='1'

METHOD_ARR=( 'AP_toppk20' )
#MODEL_ARR=( 'prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4' )
#MODEL_ARR=( 'prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4' )
SUBMETHOD_ARR=( 'a' )
P_ARR=( '0.95;0.9;0.8;0.4;0.2' )
DT_ARR=( '1.0' )

#METHOD_ARR=( 'CD_toppk20' )
#METHOD_ARR=( 'CD_topp' )
#METHOD_ARR=( 'CDk20' )
#MODEL_ARR=( 'EleutherAI/pythia-70m-deduped' )
#MODEL_ARR=( 'facebook/opt-125m' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.025;0.05;0.1;0.3;0.6' )
#P_ARR=( '0.95;0.9;0.8;0.4;0.2' )
#DT_ARR=( '0.25' )
#DT_ARR=( '0.5' )
#DT_ARR=( '1.0' )

init_existing_seeds=0
repeat_times=8


SEED_NUM=8
SEED_ARR=( 1 2 3 4 5 6 7 8 )
CUDE_IDX=( 0 1 2 3 4 5 6 7 )

#input_datasets=($(for x in "${dataset_names[@]}"; do printf "$x%.0s " {1..${repeat_times}}; done))
#for v in ${dataset_names[@]}; do for i in $(seq 1 $repeat_times); do echo $v; done; done

END=$(($init_existing_seeds + $repeat_times - 1))
input_datasets=($(for v in ${dataset_names[@]}; do for i in $(seq 1 $repeat_times); do echo $v; done; done))
existing_seeds_arr=($(seq $init_existing_seeds $END))
#existing_seeds_arr=("${existing_seeds_arr[@]}" "${existing_seeds_arr[@]}")
echo ${input_datasets[@]}
echo ${existing_seeds_arr[@]}

for j in "${!METHOD_ARR[@]}"; do
	MODEL=${MODEL_ARR[$j]}
	sample_method=${METHOD_ARR[$j]}
	sample_sub_method=${SUBMETHOD_ARR[$j]}
	top_p_all=${P_ARR[$j]}
	decay_temperature_all=${DT_ARR[$j]}
	
	final_entropy_model_path="models/$MODEL"
	batch_size=4
	#if [[ $MODEL == *"410"* ]]; then
	#	batch_size=2
	#fi
	#if [[ $MODEL == *"EleutherAI"* ]]; then
	#	batch_size=2
	#fi
	#if [[ $sample_method == *"fe_CD"* ]]; then
	#	batch_size=2
	#fi
	IFS=";" read -r -a top_p_list <<< "${top_p_all}"
	IFS=";" read -r -a decay_temperature_list <<< "${decay_temperature_all}"
	for top_p in "${top_p_list[@]}"; do
		for decay_temperature in "${decay_temperature_list[@]}"; do
			pids=()
			for i in "${!input_datasets[@]}";
			do
				dataset_name=${input_datasets[$i]}
				num_existing_seeds=${existing_seeds_arr[$i]}
				echo "python src/factual_gen/gen_fp.py --model_name=$model_name --input_file_name ${prompt_folder}/$dataset_name --cuda_idx $i --p $top_p --num_existing_seeds $num_existing_seeds --sample_method $sample_method --final_entropy_model_path $final_entropy_model_path --batch_size $batch_size --decay_temperature $decay_temperature --temperature $temperature --sample_sub_method $sample_sub_method --max_len $max_len --output_path_prefix $output_folder"
				#sleep 1 &
				python src/factual_gen/gen_fp.py --model_name=$model_name --input_file_name ${prompt_folder}/$dataset_name --cuda_idx $i --p $top_p --num_existing_seeds $num_existing_seeds --sample_method $sample_method --final_entropy_model_path $final_entropy_model_path --batch_size $batch_size --decay_temperature $decay_temperature --temperature $temperature --sample_sub_method $sample_sub_method --max_len $max_len --output_path_prefix $output_folder &
				pids+=($!)
			done
			echo "${pids[@]}"
			wait "${pids[@]}"

			MODEL_BASE=`basename ${MODEL}`

                        if [[ $sample_method == "fe_topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_fixed_${MODEL_BASE}"
                        elif [[ $sample_method == "fe_topp_period" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_our_period_r${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "fe_CD" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_alpha${top_p}_fixed_${MODEL_BASE}"
                        elif [[ $sample_method == "fe_CD_topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_fixed_${MODEL_BASE}"
                        elif [[ $sample_method == "fe_AP_topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_ait${top_p}_${entropy_model}_fixed_${MODEL_BASE}"
                        elif [[ $sample_method == "fe_topk" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_k${top_p}_fixed_${MODEL_BASE}"
                        elif [[ $sample_method == "eta" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
                        elif [[ $sample_method == "typical" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
                        elif [[ $sample_method == "topk" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}"
                        elif [[ $sample_method == "CS" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}_alpha_${decay_temperature}"
                        elif [[ $sample_method == "AP_topk" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_temp_${decay_temperature}"
                        elif [[ $sample_method == "toppk20" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_temp_${decay_temperature}"
                        elif [[ $sample_method == "decay_period" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}"
                        elif [[ $sample_method == "CD_topk" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_k${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "CD" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "CDk20" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "CD_topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "CD_toppk20" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL_BASE}"
                        elif [[ $sample_method == "AP_topa" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_alpha${top_p}_dt_${decay_temperature}_${MODEL_BASE}"
                        elif [[ $sample_method == "AP_topak20" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_alpha${top_p}_dt_${decay_temperature}_${MODEL_BASE}"
                        elif [[ $sample_method == "AP_topp" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}_${MODEL_BASE}"
                        elif [[ $sample_method == "AP_CD_toppk20" ]]; then
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}_${MODEL_BASE}"
                        elif [[ $sample_method == "AP_toppk20" ]]; then
                                #GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_${MODEL_BASE}"
                                GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}_${MODEL_BASE}"
                        fi

                        FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_${sample_method}_p${top_p}"
                        GEN_CONFIG="${PROMPT_TYPE}_${GEN_HYPER}"

			cd $eval_folder
                        echo "python src/distinct_n_each_gen.py --gen_dir ${output_folder}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT"
                        python src/distinct_n_each_gen.py --gen_dir ${output_folder}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT
                        pids=()
                        echo $GEN_CONFIG
                        #for SEED_IDX in "${SEED_ARR[@]}";
                        #do
                        for i in "${!SEED_ARR[@]}"; do
                                echo $SEED_IDX
                                SEED_IDX=${SEED_ARR[$i]}
                                PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --eval_dir ${output_folder} --final --output --num_eval_sent $NUM_EVAL_SENT
                        done
			cd -
			for i in "${!SEED_ARR[@]}"; do
                                SEED_IDX=${SEED_ARR[$i]}
                                echo "PYTHONPATH=. python src/story_gen/eval_gen.py --eval_output_folder scores_s${NUM_EVAL_SENT} --cuda_idx ${CUDE_IDX[$i]} --pred_result_path ${output_folder}/${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT &> ./temp &"
                                #export CUDA_VISIBLE_DEVICES=${CUDE_IDX[$i]}
                                PYTHONPATH=. python src/story_gen/eval_gen.py --eval_output_folder scores_s${NUM_EVAL_SENT} --cuda_idx ${CUDE_IDX[$i]} --pred_result_path ${output_folder}/${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --num_eval_sent $NUM_EVAL_SENT &> ./temp &
                                pids+=($!)
                        done
                        echo "${pids[@]}"
                        wait "${pids[@]}"
                        #unset CUDA_VISIBLE_DEVICES
		done
	done
done
