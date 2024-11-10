#!/bin/bash

FOLDER_PATH="/mnt/efs/Haw-Shiuan/true_entropy/outputs/factual_gen/"

PROMPT_TYPE="story_start2_1000"
#PROMPT_TYPE="story_1000"
#PROMPT_TYPE="story_100"

GEN_MODEL="6.9b"
#GEN_MODEL="OpenLLaMA2-7b"
#GEN_MODEL="OPT-6.7b"

#METHOD_ARR=( 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk' 'fe_topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'a' 'a' 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' 'exp_1_win' )
#P_ARR=( '1.0' '0.8;0.7;0.6;0.5;0.4;0.3' '0.1;0.3;0.8' '2.0;0.9;0.5;0.3' '0.9' '10.0;5.0;3.0;2.0;1.0' '1.0' '1.0' '1.0' )
#DT_ARR=( '0.7;1.0;1.2;1.5;2.0;3.0;4.0' '1' '1' '1' '0.95;0.9;0.7;0.5;0.3;0.1' '1' '2.0;4.0;8.0' '2.0;4.0;8.0' '0.5;1.0;1.5;2.0;4.0' )

#METHOD_ARR=( 'CD' 'topp' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' '' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'a' 'exp_real_e_only_win' 'exp_e_only_win' )
#P_ARR=( '0.6;0.4;0.2;0.1;0.05' '1.0' '1.0' '1.0' )
#DT_ARR=( '1.0' '1.0' '3.0;6.0' '3.0;6.0' )

#METHOD_ARR=( 'fe_CD' 'CD' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'pythia-70m-deduped')
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '0.4;0.6' '0.6;0.4' )
#DT_ARR=( '1.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3')
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.0;4.0;8.0' )

#METHOD_ARR=( 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.0;0.3;0.5;0.6;0.7;1.0' )
#DT_ARR=( '0.5;0.7;1.0;2.0;3.0;4.0' '1.0' )

#METHOD_ARR=( 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' )
#P_ARR=( '1.0' )
#DT_ARR=( '1.8' )

#METHOD_ARR=( 'topp' )
#MODEL_ARR=( '' )
#SUBMETHOD_ARR=( 'a' )
#P_ARR=( '0.4' )
#DT_ARR=( '1.0' )

#METHOD_ARR=( 'CD' 'fe_topp' )
#MODEL_ARR=( 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'exp_1_win' )
#P_ARR=( '0.3' '1.0' )
#DT_ARR=( '1.0' '2.0' )

METHOD_ARR=( 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
MODEL_ARR=( 'ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5_w0' '' )
#MODEL_ARR=( 'ROC_70M_bsz_128_exp_pred_last_a10_e3' '' )
SUBMETHOD_ARR=( 'exp_1_win' 'a' )
P_ARR=( '1.0' '' )
DT_ARR=( '1.2' '1.0' )
#DT_ARR=( '1.2;1.8;3.5' '1.0' )
#P_ARR=( '1.0' '0.6' )
#DT_ARR=( '2.3' '1.0' )
#P_ARR=( '1.0' '0.7' )
#DT_ARR=( '3.5' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'CD' 'fe_topp' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.3' '1.0' '0.5' )
#DT_ARR=( '2.0' '1.0' '1.8' '1.0' )
#P_ARR=( '1.0' '0.3' '1.0' '0.6' )
#DT_ARR=( '1.5' '1.0' '2.0' '1.0' )

#METHOD_ARR=( 'fe_CD_topp' 'fe_topp_period' 'topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '')
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'a' )
#P_ARR=( '1.0' '0.9' '1.0' )
#DT_ARR=( '4.0;0.7;1.5' '5.0' '0.1;0.9' )

#METHOD_ARR=( 'topp' 'CD' 'fe_topp' 'fe_topp' )
#MODEL_ARR=( '' 'pythia-70m-deduped' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3')
#SUBMETHOD_ARR=( 'a' 'a' 'exp_1_win' 'exp_e_only_win' )
#P_ARR=( '1.0' '0.25' '1.0' '1.0' )
#DT_ARR=( '0.5' '1.0' '8.0' '6.0' )

#METHOD_ARR=( 'fe_topp' 'topp' 'eta' 'typical' 'decay_period' 'topk'  )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' '' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'a' 'a' 'a' 'a' )
#P_ARR=( '1.0' '0.8' '0.8' '2.0' '0.9' '5.0'  )
#DT_ARR=( '1.2;3.0' '1' '1' '1' '0.95;0.9;0.7;0.5' '1' )

#METHOD_ARR=( 'fe_topk' 'fe_topp_period' 'typical' 'decay_period' 'topk' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' '' '' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'a' 'a' 'a' )
#P_ARR=( '10.0' '0.95' '0.1' '0.9' '1.0;3.0' )
#DT_ARR=( '4.0' '4.0' '1' '0.3;0.1' '1' )

#METHOD_ARR=( 'typical' 'fe_topp' )
#MODEL_ARR=( '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'a' 'exp_1_win' )
#P_ARR=( '0.9' '1.0' )
#DT_ARR=( '1' '0.5;1.5' )

#METHOD_ARR=( 'fe_topp_period' 'typical' 'fe_topp' )
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'a' 'exp_1_win' )
#P_ARR=( '0.9' '0.9' '1' )
#DT_ARR=( '2.0;3.0;4.0;6.0;8.0' '1' '0.5;1.5' )

#METHOD_ARR=( 'fe_topk' 'fe_topk' 'fe_topp_period' 'typical' 'fe_topp')
#MODEL_ARR=( 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' 'OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3' '' 'OWT_wiki_1e7_410M_bsz_32_exp_pred_last_a10_e3' )
#SUBMETHOD_ARR=( 'exp_1_win' 'exp_1_win' 'exp_1_win' 'a' 'exp_1_win' )
#P_ARR=( '10.0' '3.0;5.0;10.0;20.0;40.0' '0.9;0.95' '0.9' '1.0' )
#DT_ARR=( '0.5;0.7;1.0;2.0' '1.0' '2.0;3.0;4.0' '1' '0.5;1.5' )


#SEED_NUM=3
SEED_NUM=8

#TEMPERATURE=1.0

NUM_EVAL_SENT=3
#NUM_EVAL_SENT=1

top_p='1.0'

SEED_ARR=( 1 2 3 4 5 6 7 8 )

for j in "${!METHOD_ARR[@]}"; do
        MODEL=${MODEL_ARR[$j]}
        sample_method=${METHOD_ARR[$j]}
        sample_sub_method=${SUBMETHOD_ARR[$j]}
        top_p_all=${P_ARR[$j]}
        decay_temperature_all=${DT_ARR[$j]}

        IFS=";" read -r -a top_p_list <<< "${top_p_all}"
        IFS=";" read -r -a decay_temperature_list <<< "${decay_temperature_all}"
        for top_p in "${top_p_list[@]}"; do
                for decay_temperature in "${decay_temperature_list[@]}"; do
			if [[ $sample_method == "fe_topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_fixed_${MODEL}"
			elif [[ $sample_method == "fe_topp_period" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_our_period_r${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_CD" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_alpha${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_CD_topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_p${top_p}_${MODEL}"
			elif [[ $sample_method == "fe_topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_${sample_sub_method}_40_dt_${decay_temperature}_k${top_p}_${MODEL}"
			elif [[ $sample_method == "eta" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
			elif [[ $sample_method == "typical" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}"
			elif [[ $sample_method == "topk" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_k${top_p}"
			elif [[ $sample_method == "topp" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_temp_${decay_temperature}"
			elif [[ $sample_method == "decay_period" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_p${top_p}_dt_${decay_temperature}"
			elif [[ $sample_method == "CD" ]]; then
				GEN_HYPER="${GEN_MODEL}_${sample_method}_dt_${decay_temperature}_p${top_p}_${MODEL}"

			fi
			FILE_HYPER="${PROMPT_TYPE}_${GEN_MODEL}_${sample_method}_p${top_p}"
			GEN_CONFIG="${PROMPT_TYPE}_${GEN_HYPER}"
			echo "python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT"
			python src/distinct_n_each_gen.py --gen_dir ${FOLDER_PATH}/${GEN_CONFIG} --file_template ${FILE_HYPER}_gen_seed --number_of_seeds $SEED_NUM --num_eval_sent $NUM_EVAL_SENT
			pids=()
			echo $GEN_CONFIG
			#for SEED_IDX in "${SEED_ARR[@]}";
			#do
			for i in "${!SEED_ARR[@]}"; do
				echo $SEED_IDX
				SEED_IDX=${SEED_ARR[$i]}
				PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
				#PYTHONPATH=. python src/repetition.py ${GEN_CONFIG}/non${FILE_HYPER}_gen_seed${SEED_IDX}.jsonl --final --output --num_eval_sent $NUM_EVAL_SENT
			done
		done
	done
done

