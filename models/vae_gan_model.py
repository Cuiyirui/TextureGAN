import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable,grad
import util.util as util
from .base_model import BaseModel


class VaeGANModel(BaseModel):
    def name(self):
        return 'VaeGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.init_data(opt, use_D=False, use_D2=False, use_E=True, use_vae=True,use_VGGF = False, use_Dl = False)
        self.skip = False
    def is_skip(self):
        return self.skip

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        rbg_im = Variable(torch.ones(self.real_C.size()).cuda())
        self.ones_C = Variable(torch.ones(self.real_C.size(0),1,self.real_C.size(2),self.real_C.size(3)).cuda())
        self.ones_C[:,0,:,:] =  rbg_im[:,1,:,:] * 0.299 + rbg_im[:,1,:,:] * 0.587 + rbg_im[:,1,:,:] * 0.114
        # forward E
        self.mu, self.logvar = self.netE.forward(self.real_C)
        std = self.logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_encoded = eps.mul(std).add_(self.mu)

        # generate fake_B_encoded
        self.fake_C = self.netG.forward(self.ones_C, self.z_encoded)

    def encode(self, input_data):
        mu, logvar = self.netE.forward(Variable(input_data, volatile=True))
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        return eps.mul(std).add_(mu)

    def backward(self):
        kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
        self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        self.loss_L1 = self.criterionL1(self.fake_C, self.real_C)
        self.loss = self.loss_L1+self.loss_kl
        self.loss.backward()

    def update(self,data):
        self.set_input(data)
        self.forward()
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward()
        self.optimizer_E.step()
        self.optimizer_G.step()
    def is_skip(self):
        return self.skip


    def get_current_errors(self):
        z1 = self.z_encoded.data.cpu().numpy()
        loss_L1 = self.loss_L1
        ret_dict = OrderedDict([('L1_loss', loss_L1.data)])
        ret_dict['KL'] = self.loss_kl.data[0]
        return ret_dict

    def get_current_visuals(self):
        real_A_encoded = util.tensor2im(self.real_A.data)
        real_B_encoded = util.tensor2im(self.real_B.data)

        if self.opt.isTrain:
            real_C_encoded = torch.Tensor(self.real_B.size()).fill_(1.0)
            real_C_encoded[:, :, 0:self.real_C.size(2),
                 0:self.real_C.size(3)] = \
                    self.real_C.data
            real_C_encoded = util.tensor2im(real_C_encoded)

            fake_C_encoded = torch.Tensor(self.real_B.size()).fill_(1.0)
            fake_C_encoded[:, :, 0:self.fake_C.size(2),
                 0:self.fake_C.size(3)] = \
                    self.fake_C.data
            fake_C_encoded = util.tensor2im(fake_C_encoded)


            ret_dict = OrderedDict([('real_A_encoded', real_A_encoded), ('real_B_encoded', real_B_encoded),
                                        ('real_C_encoded', real_C_encoded), ('fake_C_encoded', fake_C_encoded)])
        return ret_dict

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        #if self.opt.lambda_GAN > 0.0:
            # self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netE, 'E', label, self.gpu_ids)