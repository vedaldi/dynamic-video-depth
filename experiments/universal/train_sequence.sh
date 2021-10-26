#!/bin/bash
if [ $# -lt 1 ]; then
    echo "Usage: $0 gpu "
    exit 1
fi
gpu="$1"
shift
set -e
cmd="
python train.py \
    --net scene_flow_motion_field \
    --dataset universal_sequence \
    --data_dir './checkpoints/universal/sequence/' \
    --log_time \
    --epoch_batches 2000 \
    --epoch 20 \
    --lr 1e-6 \
    --html_logger \
    --vali_batches 150 \
    --batch_size 1 \
    --optim adam \
    --vis_batches_vali 4 \
    --vis_every_vali 1 \
    --vis_every_train 1 \
    --vis_batches_train 5 \
    --vis_at_start \
    --tensorboard \
    --gpu "$gpu" \
    --save_net 1 \
    --workers 4 \
    --one_way \
    --loss_type l1 \
    --l1_mul 0 \
    --acc_mul 1 \
    --disp_mul 1 \
    --warm_sf 5 \
    --scene_lr_mul 1000 \
    --repeat 1 \
    --flow_mul 1\
    --sf_mag_div 100 \
    --time_dependent \
    --gaps 1,2,4,6,8 \
    --midas \
    --use_disp \
    --logdir './checkpoints/universal/' \
    --suffix 'track_{track_id}_{loss_type}_wreg_{warm_reg}_acc_{acc_mul}_disp_{disp_mul}_flowmul_{flow_mul}_time_{time_dependent}_CNN_{use_cnn}_gap_{gaps}_Midas_{midas}_ud_{use_disp}' \
    --test_template './experiments/universal/test_cmd.txt' \
    --force_overwrite \
    $*"
echo $cmd
eval $cmd



