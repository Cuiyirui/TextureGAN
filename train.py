# import torch.backends.cudnn as cudnn
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

total_steps = 0

for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    save_result = True
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.update_D(data)
        if model.is_skip():
            continue
        if i % opt.disc_iters == 0:
            model.update_G()
        model.balance()

        if save_result or total_steps % opt.display_freq == 0:
            save_result = save_result or total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(
                model.get_current_visuals(), epoch, ncols=1, save_result=save_result)
            save_result = False

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(
                    epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    model.update_learning_rate()

#save train result and model
import shutil
# save model
shutil.copyfile('../checkpoints_pub/contour2shirt/contour2shirt/latest_net_E.pth','./pretrained_models/latest_net_E.pth')
shutil.copyfile('../checkpoints_pub/contour2shirt/contour2shirt/latest_net_G.pth','./pretrained_models/latest_net_G.pth')
# save train result
test_name_dataset = '1K_'
# GAN loss type
if opt.GAN_loss_type == 'wGAN':
    test_name_GAN_type = 'wGAN_'
    test_name_loss_info = 'loss_clip_'+str(opt.clipping_value)+'_'
elif opt.GAN_loss_type == 'criterionGAN':
    test_name_GAN_type = 'criterionGAN_'
    test_name_loss_info = ''
# D condition
if opt.conditional_D:
    test_name_cD = 'cD'
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
            +test_name_loss_info+test_name_encode+test_name_batch+test_name_direction+'/train'
origin_name = '../checkpoints_pub/contour2shirt/contour2shirt_bicycle_gan/web'

shutil.copytree(origin_name,test_name)