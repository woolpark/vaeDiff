import os

import torch
import wandb
from dalle_pytorch import DiscreteVAE
from dalle_pytorch import OpenAIDiscreteVAE
from dalle_pytorch import VQGanVAE


class PyTorchVAEModels:
    def __init__(self):
        self.model = self.download_trained_artifact()
        self.open_vae = OpenAIDiscreteVAE().cuda()
        self.vqgan_vae = VQGanVAE().cuda()

    def download_trained_artifact(self, path="woolpark/train-vae-colab/trained-vae:v20"):
        run = wandb.init()
        artifact = run.use_artifact(path, type='model')
        artifact_dir = artifact.download()
        artifact_path = os.path.join(artifact_dir, 'vae-final.pt')
        artifact_config = artifact.metadata

        SMOOTH_L1_LOSS = artifact_config['smooth_l1_loss']
        del artifact_config['smooth_l1_loss']
        KL_LOSS_WEIGHT = artifact_config['kl_loss_weight']
        del artifact_config['kl_loss_weight']

        model_params_state = torch.load(artifact_path)
        model = DiscreteVAE(**model_params_state['hparams'],
                            smooth_l1_loss = SMOOTH_L1_LOSS,
                            kl_div_loss_weight = KL_LOSS_WEIGHT).cuda()
        model.load_state_dict(model_params_state['weights'])

        return model
