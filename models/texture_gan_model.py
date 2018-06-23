import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable,grad
import util.util as util
from .base_model import BaseModel

class TextureGANModel(BaseModel):
    def name(self):
        return 'TextureGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        use_Dl = opt.whether_local_loss
        self.init_data(opt, use_D = True, use_D2 = False, use_E = False, use_vae = False, use_VGGF = True, use_Dl = use_Dl)
        self.skip = False

    def is_skip(self):
        return self.skip

    def forward(self):
        self.skip = self.opt.isTrain and self.input_A.size(0) < self.opt.batchSize
        if self.skip:
            print('skip this point data_size = %d' % self.input_A.size(0))
            return
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        # pretrain the model
        if not self.opt.whether_local_loss:
            self.forward_pretrain()
        else:
            self.real_C = Variable(self.input_C)
            self.forward_finetuning()

    def update_D(self, data):
        self.set_requires_grad(self.netD, True)
        self.set_input(data)
        self.forward()
        if self.is_skip():
            return
        #updataD

        # pre-train process using global loss
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data, self.fake_data,
                                                         self.opt.GAN_loss_type)
        self.optimizer_D.step()
        # weight clipping if wGAN
        if self.opt.GAN_loss_type == 'wGAN':
            self.weightClipping(self.netD, self.opt.clipping_value)


        # fine-tuning process
        if self.opt.whether_local_loss:
            self.loss_Dl = 0.0
            self.optimizer_Dl.zero_grad()
            for i in range(self.real_C_blocks.size(0)):
                loss_Dl, self.losses_Dl = self.backward_D(self.netDl, self.real_C_blocks[i],
                                                          self.fake_B_blocks[i], self.opt.GAN_loss_type)
                self.loss_Dl = self.loss_Dl + loss_Dl
            self.loss_Dl = self.loss_Dl / self.real_C_blocks.size(0)
            self.optimizer_Dl.step()
            if self.opt.GAN_loss_type == 'wGAN':
                self.weightClipping(self.netDl, self.opt.clipping_value)



    def backward_D(self, netD, real, fake, loss_type):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD.forward(fake.detach())
        # real
        pred_real = netD.forward(real)
        if loss_type == 'criterionGAN':
            ## criterianGAN loss
            loss_D_fake, losses_D_fake = self.criterionGAN(pred_fake, False)
            loss_D_real, losses_D_real = self.criterionGAN(pred_real, True)
            # Combined loss
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
        elif loss_type == 'wGAN':
            ## wGAN loss
            loss_D_fake = self.wGANloss(pred_fake, False)
            loss_D_real = torch.neg(self.wGANloss(pred_real, False))
            # Combined loss
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward()
        elif loss_type == 'improved_wGAN':
            ## improved wGAN loss
            loss_D_fake = self.wGANloss(pred_fake, False)
            loss_D_real = torch.neg(self.wGANloss(pred_real, False))
            gradient_penalty = self.gradientPanelty(netD, real, fake)
            loss_D = loss_D_fake + loss_D_real + gradient_penalty
            loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def update_G(self):
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def backward_G(self):
        # 1. GAN loss
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data, self.netD, self.opt.GAN_loss_type, self.opt.lambda_GAN)

        # 2. content loss
        self.loss_c = self.criterionC(self.fake_data, self.real_data) * self.opt.lambda_c

        # 3. style loss
        self.loss_s = self.criterionS_l(self.fake_data, self.real_data) * self.opt.lambda_s

        # 4. Globel pixel loss
        self.loss_p = self.criterionL2(self.fake_data, self.real_data) * self.opt.lambda_p

        # 5. L2 color loss


        if self.opt.whether_local_loss:
            # set global style loss = 0
            self.loss_s = self.loss_s * 0
            # 6. local style loss
            self.loss_s_l = 0.0
            for i in range(self.real_C_blocks.size(0)):
                self.loss_s_l = self.loss_s_l + self.criterionS_l(self.real_C_blocks[i],
                                                                  self.fake_B_blocks[i])
            self.loss_s_l = self.loss_s_l / self.real_C_blocks.size(0)
            self.loss_s_l = self.loss_s_l * self.opt.lambda_s_l

            # 7. local gan loss
            self.loss_G_GAN_l = 0.0
            for i in range(self.fake_B_blocks.size(0)):
                self.loss_G_GAN_l = self.loss_G_GAN_l + self.backward_G_GAN(self.fake_B_blocks[i], self.netDl,
                                                                            self.opt.GAN_loss_type,
                                                                            self.opt.lambda_GAN_l)
            self.loss_G_GAN_l = self.loss_G_GAN_l / self.fake_B_blocks.size(0)
            self.loss_G_GAN_l = self.loss_G_GAN_l * self.opt.lambda_GAN_l

            # 8. local pixel loss
            if self.opt.lambda_p_l > 0.0:
                self.loss_p_l = 0.0
                for i in range(self.real_C_blocks.size(0)):
                    self.loss_p_l = self.loss_p_l + self.criterionL2(self.real_C_blocks[i], self.fake_B_blocks[i])
                self.loss_p_l = self.loss_p_l / self.real_C_blocks.size(0)
                self.loss_p_l = self.loss_p_l * self.opt.lambda_p_l
            else:
                self.loss_p_l = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_c + self.loss_s + self.loss_p
        if self.opt.whether_local_loss:
            self.loss_G = self.loss_G + self.loss_s_l + self.loss_G_GAN_l + self.loss_p_l

        self.loss_G.backward(retain_graph=True)

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
        loss_G = self.loss_G
        ret_dict = OrderedDict([('G_total', loss_G.data[0])])

        # global pixel loss
        if self.opt.lambda_p > 0.0:
            G_L2 = self.loss_p.data[0] if self.loss_p is not None else 0.0
            ret_dict['G_L2'] = G_L2

        # global gan loss
        if self.opt.lambda_GAN > 0.0:
            ret_dict['G_GAN'] = self.loss_G_GAN.data[0]
            ret_dict['D_GAN'] = self.loss_D.data[0]

        # global style loss
        if self.opt.lambda_s > 0.0:
            ret_dict['s'] = self.loss_s.data[0]

        # global content loss
        if self.opt.lambda_c > 0.0:
            ret_dict['c'] = self.loss_c.data[0]

        if self.opt.whether_local_loss:
            # local style loss
            if self.opt.lambda_s_l > 0.0:
                ret_dict['s_l'] = self.loss_s_l.data[0]
            # local gan loss
            if self.opt.lambda_GAN_l > 0.0:
                ret_dict['GAN_l'] = self.loss_G_GAN_l.data[0]
            # local pixel loss
            if self.opt.lambda_p_l > 0.0:
                ret_dict['p_l'] = self.loss_p_l.data[0]
        return ret_dict

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.isTrain:
            fake = util.tensor2im(self.fake_B.data)

        ret_dict = OrderedDict([('real_A_encoded', real_A), ('real_B_encoded', real_B),
                                ('fake_B_encoded', fake)])
        '''
        if self.opt.whether_local_loss:
            real_C_block = torch.Tensor(self.fake_B.size()).fill_(1.0)
            real_C_block[:, :, 0:self.real_C_blocks[0].size(2),
            0:self.real_C_blocks[0].size(3)] = \
                torch.unsqueeze(self.real_C_blocks[0][0].data, 0)
            real_C_block = util.tensor2im(real_C_block)

            fake_B_block = torch.Tensor(self.fake_B.size()).fill_(1.0)
            fake_B_block[:, :, 0:self.fake_B_blocks[0].size(2),
            0:self.fake_B_blocks[0].size(3)] = \
                torch.unsqueeze(self.fake_B_blocks[0][0].data, 0)
            fake_B_block_encoded = util.tensor2im(fake_B_block)
            ret_dict['real_C_block'] =s real_C_block
            ret_dict['fake_B_block'] = fake_B_block
        '''

        return ret_dict

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.opt.lambda_GAN > 0.0:
            self.save_network(self.netD, 'D', label, self.gpu_ids)



    # forward pre-train procedures
    def forward_pretrain(self):
        self.fake_B = self.netG(self.real_A)
        self.fake_data= self.fake_B
        self.real_data = self.real_B

    # forward fine-tuning procedures
    def forward_finetuning(self):
        self.fake_B = self.netG(self.real_A)
        self.fake_data = self.fake_B
        self.real_data = self.real_B
        # clip block
        self.real_C_blocks, self.fake_B_blocks = self.generate_random_block(self.real_C,self.fake_B)
