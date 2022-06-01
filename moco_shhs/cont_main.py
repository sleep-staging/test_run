#%%
import wandb
import numpy as np
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.model_selection import KFold
from data_preprocessing.new_dataloader import data_generator,cross_data_generator
from pytorch_lightning.loggers import WandbLogger
from trainer import sleep_ft,sleep_pretrain
from config import Config

config = Config()
path = '/scratch'

pretext_dir = os.path.join(path, 'SLEEP_data/wake_pretext/')
train_dir = os.path.join(path, 'SLEEP_data/train/')
test_dir = os.path.join(path, 'SLEEP_data/test/')
pretext_index = os.listdir(pretext_dir)
train_index = [os.path.join(train_dir,f_name) for f_name in os.listdir(train_dir)]
test_index = [os.path.join(test_dir,f_name) for f_name in os.listdir(test_dir)]


training_mode = 'ss'
#%%
# for self supervised training
if training_mode == 'ss':
    #name = 'with_all_subj_contraw_aug' 
    name = 'delete' 
    wandb_logger = WandbLogger(project='delete',name=name,notes='with features',save_code=True,entity='sleep-staging')
    dataloader = data_generator(pretext_dir,pretext_index,config)
    #%%
    model = sleep_pretrain(config,name,dataloader)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(auto_lr_find=True,callbacks=[lr_monitor],profiler='simple',logger=wandb_logger,max_epochs=config.num_epoch,gpus=1)
    trainer.tune(model)
    wandb_logger.watch([model],log='all',log_freq=500)
    #%%
    trainer.fit(model)
    wandb.finish()
#%%

#%%
# cross validation
else:
    file_name = 'with_all_subj_contraw_aug_trim_wake.pt'
    index= train_index + test_index
    index = np.array(index)
    kfold = KFold(n_splits=5,shuffle=True)
    name = os.path.join(config.exp_path,file_name)
    src_path = "/scratch/SLEEP_data/"
    for split,(train_ix,val_ix)  in enumerate(kfold.split(index)):
        train_idx = index[train_ix]
        val_idx = index[val_ix]
        wandb_logger = WandbLogger(project='delete',notes='with features',save_code=True,entity='sleep-staging',group="ft_linear_1_3conv_ss_"+file_name,job_type='split: '+str(split))
        train_dl,valid_dl= cross_data_generator(src_path,train_idx,val_idx,config)
        le_model = sleep_ft(name,config,train_dl,valid_dl,wandb_logger)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        le_trainer = pl.Trainer(callbacks=[lr_monitor],auto_lr_find=True,profiler='simple',logger=wandb_logger,max_epochs=config.num_ft_epoch,gpus=1)
        le_trainer.tune(le_model)
        le_trainer.fit(le_model)
        wandb.finish()
