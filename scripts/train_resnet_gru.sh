cd ../src
nohup python train.py --epochs 100 \
    --learning_rate=1e-4 \
    --weight_decay=1e-8 \
    --batch_size=256 \
    --num_workers=2 \
    --hidden_size=128 \
    --embed_size=64 \
    --num_layers=1 \
    --encoder=resnet \
    --decoder=gru \
    --run_name=002  > ../resnet_gru_run_logs.txt &