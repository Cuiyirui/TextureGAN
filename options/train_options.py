from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=4000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=50, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--lambda_weight_panelty',type=float,default=10,help='weight of gradient panelty')
        self.parser.add_argument('--which_optimizer',type=str,default='Adam',help='Types of optimizer:Adam|RMSprop ')
        # learning rate
        self.parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam:2e-4 | RMSprop:1e-4')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=100, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--disc_iters', type=int, default=1, help='number of D updates per G update')

        # pre-train lambda parameters
        self.parser.add_argument('--lambda_GAN', type=float, default=1e4, help='weight on D loss. D(G(A, E(B)))')
        self.parser.add_argument('--lambda_s', type=float, default=0, help='weight for global style loss')
        self.parser.add_argument('--lambda_p', type=float, default=1e2, help='weight for global pixel loss')
        self.parser.add_argument('--lambda_c', type=float, default=1e3, help='weight for content loss')
        self.parser.add_argument('--use_same_D', type=bool, default=True, help='if two Ds share the weights or not')
        self.isTrain = True

        # fine-tune lambda parameters
        self.parser.add_argument('--lambda_s_l', type=float, default=10, help='weight for local style loss')
        self.parser.add_argument('--lambda_p_l', type=float, default=0.01, help='weight for local pixel loss')
        self.parser.add_argument('--lambda_GAN_l', type=float, default=7e3, help='weight on local D loss')
        self.parser.add_argument('--lambda_g_l', type=float, default=0, help='weight for local glcm loss') # not used

        # content loss


        # VGG features(Style transfer)
        # self.parser.add_argument('--style_feat_layers', type=list, default=['0', '2', '5', '7', '10', '12', '14', '16', '19', '21', '23', '25', '28', '30', '32', '34'], help='feature layers for style loss')
        # self.parser.add_argument('--content_feat_layers', type=list, default=['19', '21', '23', '25'], help='feature layers for style loss')

        # local random block
        self.parser.add_argument('--block_num', type=int, default=5, help='num of random blocks')
        self.parser.add_argument('--min_block_size', type=int, default=45, help='min size of random block')
        self.parser.add_argument('--max_block_size', type=int, default=64, help='max size of random block')
        self.isTrain = True
