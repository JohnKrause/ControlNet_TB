from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
from torch.autograd import set_detect_anomaly

from torch.multiprocessing import set_sharing_strategy
from tb_dataset import TB_Dataset, TB_Dataset_distort, TB_Remote_Redis, TB_Sampler

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from config import *

def start_train():
	# Configs
	revision = REVNUM
	model_config = MODEL_CONFIG_LOCAL
	start_model = MODEL_LOCAL
	batch_size = 3
	logger_freq = 3000
	learning_rate = 30e-5
	prompt_chance = 1.0
	control_chance = 0.95
	epoch_size=25000
	max_epochs=40
	sd_locked = True
	only_mid_control = False

	set_sharing_strategy('file_system')
	set_float32_matmul_precision('medium')

	# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
	model = create_model(model_config).cpu()
	model.load_state_dict(load_state_dict(start_model, location='cpu'))
	model.learning_rate = learning_rate
	model.sd_locked = sd_locked
	model.only_mid_control = only_mid_control


	# Misc
	dataset = TB_Remote_Redis(REDIS_LIST,
						epoch_size,
						revision=revision,
						prompt_chance=prompt_chance,
						control_chance=control_chance)
	sampler = TB_Sampler(dataset, epoch_size)
	dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, sampler=sampler, pin_memory=True, persistent_workers=True)
	logger = ImageLogger(batch_frequency=logger_freq)
	trainer = pl.Trainer(accelerator='gpu', devices=2, strategy="ddp",  callbacks=[logger], accumulate_grad_batches={0:100, 4:20, 11:5, 16:1}, max_epochs=max_epochs)

	# Train!
	trainer.fit(model, dataloader)

if __name__ == '__main__':
	start_train()