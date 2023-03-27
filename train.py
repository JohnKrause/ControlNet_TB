from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
from torch.autograd import set_detect_anomaly
from tb_dataset import TB_Dataset, TB_Sampler
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from config import *


# Configs
revision = REVNUM
model_config = f"./training/{revision}/cldm_v21_v1.yaml"
start_model = f"./training/{revision}/models/{MODEL}"
batch_size = 3
logger_freq = 1000
learning_rate = 10e-5
prompt_chance = 1.0
control_chance = 0.85
epoch_size=10000
max_epochs=100
control_type = CONTROL_TYPE
sd_locked = True
only_mid_control = False

set_float32_matmul_precision('medium')

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(model_config).cpu()
model.load_state_dict(load_state_dict(start_model, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = TB_Dataset(control_type,
					revision=revision,
					prompt_chance=prompt_chance,
					control_chance=control_chance)
sampler = TB_Sampler(dataset, epoch_size)
dataloader = DataLoader(dataset, num_workers=3, batch_size=batch_size, sampler=sampler)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, callbacks=[logger], accumulate_grad_batches=3, max_epochs=max_epochs)

# Train!
trainer.fit(model, dataloader)
