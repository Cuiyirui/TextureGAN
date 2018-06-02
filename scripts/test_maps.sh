# models
RESULTS_DIR='./results/maps'
G_PATH='./pretrained_models/map2aerial_net_G.pth'
E_PATH='./pretrained_models/map2aerial_net_E.pth'

# dataset
CLASS='maps'
DIRECTION='BtoA' # from domain A to domain B
LOAD_SIZE=512 # scale images to this size
FINE_SIZE=512 # then crop to this size
INPUT_NC=3  # number of channels in the input image
ASPECT_RATIO=1.0
 # change aspect ratio for the test images

# misc
GPU_ID=0   # gpu id
HOW_MANY=10 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images


# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --G_path ${G_PATH} \
  --E_path ${E_PATH} \
  --which_direction ${DIRECTION} \
  --loadSize ${FINE_SIZE} --fineSize ${FINE_SIZE} \
  --input_nc ${INPUT_NC} \
  --how_many ${HOW_MANY} \
  --n_samples ${NUM_SAMPLES} \
  --aspect_ratio ${ASPECT_RATIO} \
  --center_crop \
  --no_flip \
  --no_encode
