source ~/.bashrc
conda activate control
CUDA_NUM="1"
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py \
				                    --base configs/v1-finetune-arcface-control-Xatt.yaml \
                                    --init_word "person" \
                                    --placeholder_string "*" \
				                    -lr 1e-6 -elr 0.005 \
                                    --gpus_num 1 --batch_size 4 \
                                    --logger_freq 2 \
                                    --save_freq 5 \
                                    --max_steps 1e9 \
                        		    --delta 1e-2 \
                        		    --opt_embed \
                                    --sample_num 10 \
                                    --debug