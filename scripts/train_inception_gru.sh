cd ../src
nohup python train.py --epochs 100 \
    --learning_rate=1e-4 \
    --weight_decay=1e-6 \
    --batch_size=128 \
    --num_workers=2 \
    --hidden_size=512 \
    --embed_size=128 \
    --num_layers=1 \
    --encoder=inception \
    --decoder=gru \
    --run_name=001  > ../inception_gru_run_logs.txt &