#%%
import torch

class Config(object):

    def __init__(self,wandb=None):
        self.input_channels = 1

        self.src_path = "/scratch/new_shhs/"
        self.exp_path = "."
        self.exp_ft_path = "/home2/vivek.talwar/sleepedf/experiment_logs/experiment/saved_ft_models"
        self.wandb = wandb
        self.batch_size = 128
        self.ft_mod = 'eeg'

        self.dropout = 0.2
        self.features_len  = 190 

        self.degree = 0.05
        self.permutation_segments = 3
        self.permutation_mode = "random"
        self.mask_max_points = 200
        self.mask_min_points = 50 

        self.final_out_channels = 128

        # time domain resnet parameters
        self.kernel_size_stem =7 
        self.stride_stem=2
        self.kernel_size = 5
        self.stride=2
        self.tc_hidden_dim = 128
        self.in_channels = 1
        self.num_classes = 5

        self.dc_max = 5 
        self.dc_min = -5

        # loss parameters
        self.temperature = 0.5
        self.use_cosine_similarity = True

        # optimizer paramters
        self.optimizer = "adam"
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 0.0003

        self.num_epoch = 200
        self.num_ft_epoch = 70

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drop_last = True

        self.nperseg = 250
        self.noverlap = 200
