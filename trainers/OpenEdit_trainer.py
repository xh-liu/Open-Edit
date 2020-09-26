from models.OpenEdit_model import OpenEditModel
import torch

class OpenEditTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.open_edit_model = OpenEditModel(opt)
        if opt.mpdist:
            self.open_edit_model = torch.nn.parallel.DistributedDataParallel(self.open_edit_model, device_ids=[opt.gpu], find_unused_parameters=True)
            self.open_edit_model_on_one_gpu = self.open_edit_model.module
        else:
            self.open_edit_model_on_one_gpu = self.open_edit_model

        self.generated = None
        self.loss_gan = not opt.no_disc
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.open_edit_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.open_edit_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.open_edit_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses
