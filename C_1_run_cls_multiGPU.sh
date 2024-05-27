#chmod +x C_1_run_cls_multiGPU.sh

set -ex
export CUDA_VISIBLE_DEVICES=0,1

max_epoch="10"

# sliding window
sw=0
window=512

#model_type='bertweet'
#model_name='vinai/bertweet-base'
#alias='bt'

model_type='roberta'
model_name='roberta-base'
alias='rb'

proj=migraine_sentiment/basic

for k in 1 2 3 4 5
do
	data_dir=./data/${proj}/data_splits_${k}
	out_dir=./model/${proj}/data_splits_${k}_${alias}

	if [ ! -d ${out_dir} ];
	then
		mkdir -p ${out_dir}
		python C_2_simpletransformers_cls.py \
			--data_dir ${data_dir} \
			--out_dir ${out_dir} \
			--epoch ${max_epoch} \
			--sliding_window $sw \
			--max_seq_length $window \
			--model_type ${model_type} \
			--model_name ${model_name} \
			--train_batch_size 16 \
			--eval_batch_size 16 \
			--gradient_accumulation_steps 2 \
			--n_gpu 2 \
			--do_train --do_predict
			
	fi
done
