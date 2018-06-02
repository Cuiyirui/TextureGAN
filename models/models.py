def create_model(opt):
    model = None
    print('Loading model %s...' % opt.model)

    if opt.model == 'bicycle_gan':
        from .bicycle_gan_model import BiCycleGANModel
        model = BiCycleGANModel()
    elif opt.model =='cloth_gan_2':
        from .cloth_gan_2_model import ClothGAN2Model
        model = ClothGAN2Model()
    elif opt.model =='vae':
        from .vae_gan_model import VaeGANModel
        model = VaeGANModel()
    elif opt.model =='texture_gan':
        from .texture_gan_model import TextureGANModel
        model = TextureGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
