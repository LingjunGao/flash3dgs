SCENE_DIR="data/360_v2"
CKPT_DIR="results/benchmark"
RESULT_DIR="results/test_461"
TYPE="mlp"
SCENE_LIST="bicycle" # treehill flowers bicycle garden stump bonsai counter kitchen room
RENDER_TRAJ_PATH="ellipse"


echo "$RESULT_DIR"
for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4 #4
    fi
    
    echo "Running $SCENE"

    # run eval and render
    for CKPT in $CKPT_DIR/$SCENE/ckpts/ckpt_6999_rank0.pt;
    do
       CUDA_LAUNCH_BLOCKING=1  CUDA_VISIBLE_DEVICES=0 python load_simple_trainer.py default --disable_viewer --type $TYPE --data_factor $DATA_FACTOR \
            --render_traj_path $RENDER_TRAJ_PATH \
            --data_dir data/360_v2/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE\
            --ckpt $CKPT\
            ## --pure_eval
    done
done


