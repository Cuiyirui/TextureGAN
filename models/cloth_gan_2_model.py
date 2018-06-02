import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable,grad
import util.util as util
from .base_model import Stage2BaseModel


class ClothGAN2Model(Stage2BaseModel):
    def name(self):
        return 'ClothGAN2Model'
    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batchSize % 2 == 0  # load two images at one time.
        # D is used to generate texture that close to ground truth
        use_D = opt.isTrain and opt.lambda_GAN_l > 0.0
        use_E = opt.isTrain
        Stage2BaseModel.initialize(self, opt)
        self.init_data(opt,use_vae=True)
        self.skip = False

    def is_skip(self):
        return self.skip

    def forward(self):
        # get real images
        self.skip = self.opt.isTrain and self.input_A.size(0) < self.opt.batchSize
        if self.skip:
            print('skip this point data_size = %d' % self.input_A.size(0))
            return
        half_size = self.opt.batchSize // 2
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        # A is contour image B is ground truth
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]

        if self.opt.weather_random:
            self.real_A_random = self.real_A[half_size:]
            self.real_B_random = self.real_B[half_size:]
            self.real_C_random = self.real_C[half_size:]
        # whether clip cloth
        if self.opt.whether_encode_cloth:
            self.real_C = Variable(self.input_C)
            self.real_C_encoded = self.real_C[0:half_size]
        if self.opt.whether_encode_cloth and self.opt.which_image_encode == 'groundTruth':
            self.forward_AtoBencodeC()

    def forward_AtoBencodeC(self):
        # clip index
        clip_start_index = (self.opt.fineSize - self.opt.encode_size) // 2
        clip_end_index = clip_start_index + self.opt.encode_size

        # compute latent code of clipped cloth
        self.mu, self.logvar = self.netE.forward(self.real_C_encoded)
        std = self.logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_encoded = eps.mul(std).add_(self.mu)

        # generate fake_B_encoded
        self.fake_B_encoded = self.netG.forward(self.real_A_encoded, self.z_encoded)

        # generate random blocks between fake image and ground truth
        self.real_B_encoded_cloths, self.fake_B_encoded_cloths = self.generate_random_block(self.real_C_encoded,self.fake_B_encoded)

        if self.opt.weather_random:
            # get random z
            self.z_random = self.get_z_random(self.real_A_random.size(0), self.opt.nz, 'gauss')

            # generate fake_B_random and clip a cloth
            self.fake_B_random = self.netG.forward(self.real_A_encoded, self.z_random)

            # generate random blocks between random fake image and ground truth
            self.real_B_random_cloths, self.fake_B_random_cloths = self.generate_random_block(self.real_B_random, self.fake_B_random)

        if self.opt.conditional_D:  # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            if self.opt.weathe_random:
                self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
                self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded_cloths
            self.real_data_encoded = self.real_B_encoded_cloths
            if self.opt.weather_random:
                self.fake_data_random = self.fake_B_random_cloths
                self.real_data_random = self.real_B_random_cloths

        if self.opt.lambda_z > 0.0:
            self.fake_C_random = self.fake_B_random[:, :, clip_start_index:clip_end_index, clip_start_index:clip_end_index]
            self.mu2, logvar2 = self.netE.forward(self.fake_C_random)  # mu2 is a point estimate

    def encode(self, input_data):
        mu, logvar = self.netE.forward(Variable(input_data, volatile=True))
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        return eps.mul(std).add_(mu)

    def backward_D(self, netD, real, fake, loss_type):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD.forward(fake.detach())
        # real
        pred_real = netD.forward(real)
        if loss_type=='criterionGAN':
            ## criterianGAN loss
            loss_D_fake, losses_D_fake = self.criterionGAN(pred_fake, False)
            loss_D_real, losses_D_real = self.criterionGAN(pred_real, True)
            # Combined loss
            loss_D = loss_D_fake + loss_D_real
            loss_D = self.opt.lambda_GAN_D*loss_D
            loss_D.backward()
        elif loss_type=='wGAN':
            ## wGAN loss
            loss_D_fake = self.wGANloss(pred_fake,False)
            loss_D_real = torch.neg(self.wGANloss(pred_real,False))
            # Combined loss
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
        elif loss_type=='improved_wGAN':
            ## improved wGAN loss
            loss_D_fake = self.wGANloss(pred_fake,False)
            loss_D_real = torch.neg(self.wGANloss(pred_real,False))
            gradient_penalty = self.gradientPanelty(netD, real, fake)
            loss_D = loss_D_fake + loss_D_real + gradient_penalty
            loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, loss_type='criterionGAN' ,ll=0.0):
        if ll > 0.0:
            pred_fake = netD.forward(fake)
            if loss_type == 'criterionGAN':
                loss_G_GAN, losses_G_GAN = self.criterionGAN(pred_fake, True)
            elif loss_type == 'wGAN' or loss_type == 'improved_wGAN':
                loss_G_GAN = self.wGANloss(pred_fake, False)

        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded[0], self.netD, self.opt.GAN_loss_type, self.opt.lambda_GAN_l)
        if self.opt.weather_random:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random[0], self.netD, self.opt.GAN_loss_type, self.opt.lambda_GAN2_l)
        else:
            self.loss_G_GAN2 = self.loss_G_GAN*0
        # 2. KL loss
        kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl

        # 3. reconstruction |fake_B-real_B|
        self.loss_G_L1 = self.criterionL1(self.fake_data_encoded[0], self.real_data_encoded[0]) * self.opt.lambda_L1

        # 4. style loss
        self.loss_s_l = self.criterionS_l(self.fake_data_encoded[0],self.real_data_encoded[0]) * self.opt.lambda_s_l

        # 5. gram loss
        self.loss_g_l = self.criterionGLCM(self.fake_data_encoded[0],self.real_data_encoded[0])* self.opt.lambda_g_l

        # 6. reconstruction |fake_B-real_B|
        self.loss_c = self.criterionC(self.fake_data_encoded[0], self.real_data_encoded[0]) * self.opt.lambda_c

        # 7. local pixel loss
        self.loss_p_l = self.criterionL2(self.fake_data_encoded[0],self.real_data_encoded[0]) * self.opt.lambda_p_l


        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_kl + self.loss_G_L1 + self.loss_c + self.loss_s_l + self.loss_p_l + self.loss_g_l
        self.loss_G.backward(retain_graph=True)

    def update_D(self, data):
        self.set_requires_grad(self.netD, True)
        self.set_input(data)
        self.forward()
        if self.is_skip():
            return
        # update D1
        if self.opt.lambda_GAN_l > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded[0], self.fake_data_encoded[0],self.opt.GAN_loss_type)
            if self.opt.weather_random:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random[0], self.fake_data_random[0],self.opt.GAN_loss_type)
            self.optimizer_D.step()
            # weight clipping if wGAN
            if self.opt.GAN_loss_type=='wGAN':
                self.weightClipping(self.netD,self.opt.clipping_value)

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0 and not self.opt.which_image_encode == 'contour':
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        # weight clipping if wGAN and clipping G
        if self.opt.GAN_loss_type == 'wGAN' and self.opt.whether_clipping_G:
            self.weightClipping(self.netG, self.opt.G_clipping_value)
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0 and not self.opt.which_image_encode == 'contour':
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()


    def wGAN_loss(self, input_data):
        losses = torch.zeros(np.shape(input_data)[0])
        for idx, single_data in enumerate(input_data):
            patch_loss = torch.mean(single_data.data)
            losses[idx] = patch_loss
        mean_loss = torch.FloatTensor(1).fill_(torch.mean(losses))
        return Variable(mean_loss, requires_grad=False).cuda()

    def weightClipping(self, net, clipping_value):
        for p in net.parameters():
            p.data.clamp_(-clipping_value, clipping_value)

    def gradientPanelty(self, netD, real_data, fake_data):
        bz = self.opt.batchSize / 2
        alpha = torch.rand(bz, 1)
        alpha = alpha.expand(bz, int(real_data.nelement() / bz)).contiguous()
        alpha = alpha.view(bz, real_data.size(1), real_data.size(2), real_data.size(3))
        alpha = alpha.cuda()

        interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)
        # patchGAN loss
        disc_interpolates = torch.mean(disc_interpolates[0]) + torch.mean(disc_interpolates[1])

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_weight_panelty
        return gradient_penalty

    def get_current_errors(self):
        z1 = self.z_encoded.data.cpu().numpy()
        if self.opt.lambda_z > 0.0 and not self.opt.which_image_encode == 'contour':
            loss_G = self.loss_G + self.loss_z_L1
        else:
            loss_G = self.loss_G
        ret_dict = OrderedDict([('z_encoded_mag', np.mean(np.abs(z1))),
                                ('G_total', loss_G.data[0])])

        if self.opt.lambda_L1 > 0.0:
            G_L1 = self.loss_G_L1.data[0] if self.loss_G_L1 is not None else 0.0
            ret_dict['G_L1_encoded'] = G_L1

        if self.opt.lambda_z > 0.0 and not self.opt.which_image_encode == 'contour':
            z_L1 = self.loss_z_L1.data[0] if self.loss_z_L1 is not None else 0.0
            ret_dict['z_L1'] = z_L1

        if self.opt.lambda_kl > 0.0:
            ret_dict['KL'] = self.loss_kl.data[0]

        if self.opt.lambda_GAN_l > 0.0:
            ret_dict['G_GAN'] = self.loss_G_GAN.data[0]
            ret_dict['D_GAN'] = self.loss_D.data[0]
            # ret_dict['D_GAN_fake'] = self.losses_D[0]
            # ret_dict['D_GAN_real'] = self.losses_D[1]

        if self.opt.lambda_GAN2_l > 0.0:
            ret_dict['G_GAN2'] = self.loss_G_GAN2.data[0]
            ret_dict['D_GAN2'] = self.loss_D2.data[0]

        if self.opt.lambda_s_l > 0.0:
            ret_dict['s_l'] = self.loss_s_l.data[0]

        if self.opt.lambda_p_l > 0.0:
            ret_dict['p_l'] = self.loss_p_l.data[0]

        if self.opt.lambda_g_l > 0.0:
            ret_dict['g_l'] = self.loss_g_l.data[0]

        if self.opt.lambda_c > 0.0:
            ret_dict['C'] = self.loss_c.data[0]

        return ret_dict

    def get_current_visuals(self):
        real_A_encoded = util.tensor2im(self.real_A_encoded.data)
        real_B_encoded = util.tensor2im(self.real_B_encoded.data)

        if self.opt.weather_random:
            real_A_random = util.tensor2im(self.real_A_random.data)
            real_B_random = util.tensor2im(self.real_B_random.data)
            fake_random = util.tensor2im(self.fake_B_random.data)

        if self.opt.isTrain:
            fake_encoded = util.tensor2im(self.fake_B_encoded.data)
            if self.opt.whether_encode_cloth:
                real_C_encoded = torch.Tensor(self.fake_B_encoded.size()).fill_(1.0)
                real_C_encoded[:, :, 0:self.real_C_encoded.size(2),
                0:self.real_C_encoded.size(3)] = \
                    self.real_C_encoded.data
                real_C_encoded = util.tensor2im(real_C_encoded)

                if self.opt.weather_random:
                    ret_dict = OrderedDict([('real_A_encoded', real_A_encoded), ('real_B_encoded', real_B_encoded),
                                            ('real_C_encoded', real_C_encoded), ('fake_B_encoded', fake_encoded),
                                            ('fake_B_random', fake_random)])
                else:
                    ret_dict = OrderedDict([('real_A_encoded', real_A_encoded), ('real_B_encoded', real_B_encoded),
                                            ('real_C_encoded', real_C_encoded), ('fake_B_encoded', fake_encoded)])
                # if use z_L1
                if self.opt.lambda_z > 0.0 and not self.opt.which_image_encode == 'contour':
                    fake_C_random = torch.Tensor(self.fake_B_random.size()).fill_(1.0)
                    fake_C_random[:, :, 0:self.fake_C_random.data.size(2),
                    0:self.fake_C_random.data.size(3)] = \
                        self.fake_C_random.data
                    fake_C_random = util.tensor2im(fake_C_random)
                    ret_dict['fake_C_random'] = fake_C_random

                # if use local loss
                if self.opt.whether_local_loss:
                    real_C_block_encoded = torch.Tensor(self.fake_B_random.size()).fill_(1.0)
                    real_C_block_encoded[:, :, 0:self.real_C_blocks_encoded[0].size(2),
                    0:self.real_C_blocks_encoded[0].size(3)] = \
                        torch.unsqueeze(self.real_C_blocks_encoded[0][0].data, 0)
                    real_C_block_encoded = util.tensor2im(real_C_block_encoded)

                    fake_B_block_encoded = torch.Tensor(self.fake_B_random.size()).fill_(1.0)
                    fake_B_block_encoded[:, :, 0:self.fake_B_blocks_encoded[0].size(2),
                    0:self.fake_B_blocks_encoded[0].size(3)] = \
                        torch.unsqueeze(self.fake_B_blocks_encoded[0][0].data, 0)
                    fake_B_block_encoded = util.tensor2im(fake_B_block_encoded)

                    ret_dict['real_C_block_encoded'] = real_C_block_encoded
                    ret_dict['fake_B_block_encoded'] = fake_B_block_encoded
            else:
                ret_dict = OrderedDict([('real_A_encoded', real_A_encoded), ('real_B_encoded', real_B_encoded),
                                        ('real_A_random', real_A_random), ('real_B_random', real_B_random),
                                        ('fake_random', fake_random), ('fake_encoded', fake_encoded)])
            return ret_dict

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.opt.lambda_GAN_l > 0.0:
            self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.lambda_GAN2_l > 0.0:
            self.save_network(self.netD, 'D2', label, self.gpu_ids)
        if self.opt.whether_local_loss and self.opt.lambda_GAN_l > 0.0:
            self.save_network(self.netDl, 'Dl', label, self.gpu_ids)
        self.save_network(self.netE, 'E', label, self.gpu_ids)

