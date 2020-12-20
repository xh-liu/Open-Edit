import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
import pickle
from models.networks.txt_enc import EncoderText
from models.networks.perturbation import PerturbationNet
import random

def l2norm(x, norm_dim=1):
    norm = torch.pow(x, 2).sum(dim=norm_dim, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class OpenEditModel(torch.nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor
        self.perturbation = opt.perturbation

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        self.generator = opt.netG

        if self.perturbation:
            self.netP = PerturbationNet(opt)
            self.netP.cuda()
        if self.opt.manipulation:
            self.vocab = pickle.load(open(opt.vocab_path, 'rb'))
            self.txt_enc = EncoderText(len(self.vocab), 300, 1024, 1)
            self.txt_enc.load_state_dict(torch.load(opt.vse_enc_path, map_location='cpu')['model'][1])
            self.txt_enc.eval().cuda()

        # set loss functions
        if self.perturbation:
            self.criterionPix = torch.nn.L1Loss()
            self.criterionVGG = networks.VGGLoss(self.opt.gpu)
        elif opt.isTrain:
            self.loss_l1pix = opt.l1pix_loss
            self.loss_gan = not opt.no_disc
            self.loss_ganfeat = not opt.no_ganFeat_loss
            self.loss_vgg = not opt.no_vgg_loss

            if self.loss_l1pix:
                self.criterionPix = torch.nn.L1Loss()
            if self.loss_gan:
                self.criterionGAN = networks.GANLoss(
                    opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if self.loss_ganfeat:
                self.criterionFeat = torch.nn.L1Loss()
            if self.loss_vgg:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu)

    def forward(self, data, mode, ori_cap=None, new_cap=None, alpha=1, global_edit=False):
        if not data['image'].is_cuda:
            data['image'] = data['image'].cuda()
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(data)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(data['image'])
            return fake_image
        elif mode == 'manipulate':
            with torch.no_grad():
                fake_image = self.manipulate(data['image'], ori_cap, new_cap, alpha, global_edit=global_edit)
            return fake_image
        elif mode == 'optimize':
            g_loss, generated = self.optimizeP(data, ori_cap, new_cap, alpha, global_edit=global_edit)
            return g_loss, generated
        else:
            raise ValueError("|mode| is invalid")

    def create_P_optimizers(self, opt):
        P_params = list(self.netP.parameters())

        beta1, beta2 = 0, 0.9
        P_lr = opt.lr

        optimizer_P = torch.optim.Adam(P_params, lr=P_lr, betas=(beta1, beta2))

        return optimizer_P


    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain and self.loss_gan:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        if self.loss_gan and opt.isTrain:
            optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))
        else:
            optimizer_D = None

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        if self.loss_gan:
            util.save_network(self.netD, 'D', epoch, self.opt)

    def initialize_networks(self, opt):
        print(opt.isTrain)
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain and not opt.no_disc else None
        netE = networks.define_E(opt)

        if not opt.isTrain or opt.continue_train or opt.manipulation:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain or opt.needs_D:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
                print('network D loaded')

        return netG, netD, netE

    def compute_generator_loss(self, data):
        G_losses = {}

        fake_image, spatial_embedding = self.generate_fake(data['image'])

        if self.loss_l1pix:
            G_losses['Pix'] = self.criterionPix(fake_image, data['image'])
        if self.loss_gan:
            pred_fake, pred_real = self.discriminate(fake_image, data)

        if self.loss_ganfeat:
            actual_num_D = 0
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                if num_intermediate_outputs == 0:
                    continue
                else:
                    actual_num_D += 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat
            G_losses['GAN_Feat'] = GAN_Feat_loss / actual_num_D

        if self.loss_gan:    
            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)

        if self.loss_vgg:
            G_losses['VGG'] = self.criterionVGG(fake_image, data['image']) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, data):
        D_losses = {}
        with torch.no_grad():
            fake_image, spatial_embedding = self.generate_fake(data['image'])

        pred_fake, pred_real = self.discriminate(fake_image, data)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def generate_fake(self, real_image, return_skip = False, return_edge = False):

        with torch.no_grad():
            spatial_embedding, edge = self.netE(real_image)
        fake_image = self.netG(spatial_embedding, edge)
        if return_edge:
            return fake_image, spatial_embedding, edge
        else:
            return fake_image, spatial_embedding

    def generate_fake_withP(self, real_image):
        with torch.no_grad():
            spatial_embedding, edge = self.netE(real_image)
        p, P_reg = self.netP()
        fake_image = self.netG(spatial_embedding, edge, perturbation=True, p=p)
        
        return fake_image, P_reg

    def discriminate(self, fake_image, data, edge=None):

        fake_and_real_img = torch.cat([fake_image, data['image']], dim=0)

        discriminator_out = self.netD(fake_and_real_img)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def manipulate(self, real_image, ori_txt, new_txt, alpha, global_edit=False):
        
        spatial_embedding, edge = self.netE(real_image, norm=True)

        with torch.no_grad():
            ori_txt = self.txt_enc(ori_txt, [ori_txt.shape[1]])
            new_txt = self.txt_enc(new_txt, [new_txt.shape[1]])

        proj = spatial_embedding * ori_txt.unsqueeze(2).unsqueeze(3).repeat(1,1,spatial_embedding.shape[2],spatial_embedding.shape[3])
        proj_s = proj.sum(1, keepdim=True)
        proj = proj_s.repeat(1,1024,1,1)
        # proj = F.sigmoid(proj)

        if global_edit:
            proj[:] = 1 # for global attributes, don't need to do grounding

        spatial_embedding = spatial_embedding - alpha * proj * ori_txt.unsqueeze(2).unsqueeze(3).repeat(1,1,spatial_embedding.shape[2],spatial_embedding.shape[3])
        spatial_embedding = spatial_embedding + alpha * proj * new_txt.unsqueeze(2).unsqueeze(3).repeat(1,1,spatial_embedding.shape[2],spatial_embedding.shape[3])

        spatial_embedding = l2norm(spatial_embedding)
        
        if self.perturbation:
            p, P_reg = self.netP()
            fake_image = self.netG(spatial_embedding, edge, perturbation=True, p=p)
            return fake_image
        else:
            fake_image = self.netG(spatial_embedding, edge)
            return fake_image

    def optimizeP(self, data, ori_cap, new_cap, alpha, global_edit=False):

        P_losses = {}

        fake_image, P_reg = self.generate_fake_withP(data['image'])

        manip_image = self.manipulate(data['image'], ori_cap, new_cap, alpha, global_edit=global_edit)
        cycle_image = self.manipulate(manip_image, new_cap, ori_cap, alpha, global_edit=global_edit)

        P_losses['Pix'] = self.criterionPix(fake_image, data['image'])
        P_losses['VGG'] = self.criterionVGG(fake_image, data['image']) * self.opt.lambda_vgg
        
        P_losses['cyclePix'] = self.criterionPix(cycle_image, data['image'])
        P_losses['cycleVGG'] = self.criterionVGG(cycle_image, data['image']) * self.opt.lambda_vgg

        P_losses['Reg'] = P_reg

        return P_losses, fake_image
