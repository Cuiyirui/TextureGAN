import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import numpy as np


# helper function
def get_random_z(opt):
    z_samples = np.random.normal(0, 1, (opt.n_samples + 1, opt.nz))
    return z_samples


# options
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads=1
opt.batchSize = 1   # test code only supports batchSize=1
opt.serial_batches = True  # no shuffle
# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase +
                       '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, G = %s, E = %s' % (
    opt.name, opt.phase, opt.G_path, opt.E_path))

# sample random z
if opt.sync:
    z_samples = get_random_z(opt)

# test stage
for i, data in enumerate(islice(dataset, opt.how_many)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.how_many))
    if not opt.sync:
        z_samples = get_random_z(opt)
    for nn in range(opt.n_samples + 1):
        encode_B = nn == 0 and not opt.no_encode
        _, real_A, fake_B, real_B = model.test_simple()
        if nn == 0:
            all_images = [real_A, real_B, fake_B]
            all_names = ['input', 'ground truth', 'encoded']
            #from skimage import io
            #io.imshow(real_A)
        else:
            all_images.append(fake_B)
            all_names.append('random sample%2.2d' % nn)

    img_path = 'input image%3.3i' % i
    save_images(webpage, all_images, all_names, img_path, None,
                width=opt.fineSize, aspect_ratio=opt.aspect_ratio)

webpage.save()
'''
#save test result
test_name_dataset = '1K_'
if opt.GAN_loss_type == 'wGAN':
    test_name_GAN_type = 'wGAN_'
    test_name_loss_info = 'loss_clip_'+str(opt.clipping_value)+'_'
elif opt.GAN_loss_type == 'criterionGAN':
    test_name_GAN_type = 'criterionGAN_'
    test_name_loss_info = ''
# D condition
if opt.conditional_D:
    test_name_cD = 'cD_'
else:
    test_name_cD = '_'
# local loss
if opt.whether_local_loss:
    test_name_local = 'local_'
else:
    test_name_local = '_'
test_name_encode = 'encode_'+ str(opt.encode_size)+'_'
test_name_batch = 'batch_'+ str(opt.batchSize)+'_'
test_name_direction = 'direction_'+opt.which_direction

test_name = '../results/results_backup/'+test_name_dataset+test_name_GAN_type+test_name_local+test_name_cD\
            +test_name_loss_info+test_name_encode+test_name_batch+test_name_direction+'/'+opt.phase
origin_name = './results/edges_cloth2shirt/'+opt.phase
import shutil
shutil.copytree(origin_name,test_name)
'''