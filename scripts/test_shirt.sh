# models
RESULTS_DIR='./results/edges_cloth2shirt'
G_PATH='./pretrained_models/latest_net_G.pth'
E_PATH='./pretrained_models/latest_net_E.pth'

# dataset
CLASS='contour2shirt'
DIRECTION='BtoA' # from domain A to domain B
LOAD_SIZE=256 # scale images to this size
FINE_SIZE=256 # then crop to this size
INPUT_NC=1  # number of channels in the input image

# misc
GPU_ID=1   # gpu id
HOW_MANY=50 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
# CUDA_VISIBLE_DEVICES=${GPU_ID} 
python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --center_crop \
  --no_flip \
  ${1}
