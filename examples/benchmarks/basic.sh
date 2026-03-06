#!/bin/bash
SCENE_DIR="data/360_v2"
RESULT_DIR="results/test/test"
TYPE="cuda"
SCENE_LIST="bicycle bonsai counter garden kitchen room stump" #  bicycle bonsai counter garden kitchen room stump 
RENDER_TRAJ_PATH="ellipse"



for ((i = 1; i <= 30; ++i));
do
for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4 #4
    fi
    
    echo "Running $SCENE"


    # run eval and render
    for CHECK in results/benchmark/$SCENE/ckpts/*;
    do
    # CHECK="results/benchmark/$SCENE/ckpts/ckpt_29999_rank0.pt"
       CUDA_LAUNCH_BLOCKING=1  CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable_viewer  --data_factor $DATA_FACTOR \
            --ckpt $CHECK \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/\
            ## --pure_eval
    done
done
done

