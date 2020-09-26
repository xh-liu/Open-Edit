from models.OpenEdit_model import OpenEditModel
import torch
import torch.nn as nn

class OpenEditOptimizer():
    """
    Optimizer perturbation parameters during testing
    """

    def __init__(self, opt):
        self.opt = opt
        self.open_edit_model = OpenEditModel(opt)
        self.open_edit_model_on_one_gpu = self.open_edit_model

        self.generated = None

        self.optimizer_P = self.open_edit_model_on_one_gpu.create_P_optimizers(opt)
        self.old_lr = opt.lr

    def run_opt_one_step(self, data, ori_cap, new_cap, alpha, global_edit=False):
        self.optimizer_P.zero_grad()
        r_losses, generated = self.open_edit_model(
            data, mode='optimize', ori_cap=ori_cap, new_cap=new_cap, 
            alpha=alpha, global_edit=global_edit)
        r_loss = sum(r_losses.values()).mean()
        r_loss.backward()
        self.optimizer_P.step()
        self.r_losses = r_losses
        self.generated = generated

    def get_latest_losses(self):
        return {**self.r_losses}

    def get_latest_generated(self):
        return self.generated
