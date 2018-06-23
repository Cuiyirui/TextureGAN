<img src='imgs/day2night.gif' align="right" width=360>

<br><br><br><br>

# TextureGAN
This is the our implementation version of CVPR2018 paper TextureGAN.
# Train
Run train.py

The training options is:
--dataroot
./datasets/contour2shirt
--checkpoints_dir
../checkpoints_pub/textureGAN
--loadSize
256
--fineSize
256
--nz
8
--input_nc
3
--niter
200
--niter_decay
200
--use_dropout

# Test
Run test.py

The test options is:
--dataroot
./datasets/contour2shirt/
--results_dir
./results/edges_cloth2shirt
--G_path
./pretrained_models/latest_net_G.pth
--which_direction
AtoB
--loadSize
256
--fineSize
256
--input_nc
3
--how_many
50
--n_samples
10
--center_crop
--no_flip
