mkdir 'log'
python train.py \
    --data_root 'D:/BIMO/Programming/worklab/TA/dataset/data' \
    --train_file 'D:/BIMO/Programming/worklab/TA/dataset/data/complete_dataset.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'ArcFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 10 \
    --save_freq 1000 \
    --batch_size 96 \
    --momentum 0.9 \
    --log_dir 'log' \
    --tensorboardx_logdir 'arc_MobileFaceNet' \
    2>&1 | tee log/log.log
