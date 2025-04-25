import torch
import itertools 
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import init_weights


class SineCycleGanModel(BaseModel):
    """
    CycleGAN for 1D sine↔noisy translation.

    A: clean sine windows ([B,1,L])
    B: noisy sine windows ([B,1,L])
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='dummy', no_dropout=True) 
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0,
                                help='weight for cycle loss A→B→A (denoise→noise→denoise)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss B→A→B (noise→denoise→noise)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='identity loss weight')

        return parser

    def __init__(self, opt):
        """Initialize generators G_A: A→B and G_B: B→A, plus discriminators D_A, D_B."""
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A',
                           'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A',
                             'real_B', 'fake_A', 'rec_B']
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.Generator1D(
            in_ch=1, out_ch=1, 
            n_blocks=opt.n_blocks,
            ngf=opt.ngf
        ).to(self.device)
        self.netG_B = networks.Generator1D(
            in_ch=1, out_ch=1,
            n_blocks=opt.n_blocks,
            ngf=opt.ngf
        ).to(self.device)

        if self.isTrain:
            self.netD_A = networks.Discriminator1D(in_ch=1, ndf=opt.ndf).to(self.device)
            self.netD_B = networks.Discriminator1D(in_ch=1, ndf=opt.ndf).to(self.device)

            # image pools to reduce model oscillation
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # loss functions 
            self.criterionGAN = networks.GANLoss(gan_mode=opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            # optimizers 
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )

            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)
            )
            self.optimizers = [self.optimizer_G, self.optimizer_D]

        init_weights(self.netG_A, opt.init_type, opt.init_gain)
        init_weights(self.netG_B, opt.init_type, opt.init_gain)
        if self.isTrain:
            init_weights(self.netD_A, opt.init_type, opt.init_gain)
            init_weights(self.netD_B, opt.init_type, opt.init_gain)

    def set_input(self, input):
        self.real_A = input['data_A'].to(self.device) # clean sine  
        self.real_B = input['data_B'].to(self.device) # noisy sine
        self.image_paths = input['paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A) → B
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A)) → A
        self.fake_A = self.netG_B(self.real_B)  # G_B(B) → A
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B)) → B

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Computes generator loss: adversarial loss, cycle loss and identity loss."""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt 
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt 
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0 

        # gan losses 
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # cycle consistency losses 
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B 

        # combine losses 
        self.loss_G = (
            self.loss_G_A + self.loss_G_B + 
            self.loss_cycle_A + self.loss_cycle_B +
            self.loss_idt_A + self.loss_idt_B
        )

        self.loss_G.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # D_A and D_B require no gradients when optimizing G_A and G_B
        self.optimizer_G.zero_grad() 
        self.backward_G()            # calculate gradients for G_A and G_B
        self.optimizer_G.step()          # update G_A and G_B

        # discriminators 
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()          # calculate gradients for D_A 
        self.backward_D_B()          # calculate graidents for D_B
        self.optimizer_D.step()          # update D_A and D_B
