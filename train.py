from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tb_dataset import TB_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from config import *


# Configs
revision = REVNUM
model_config = f"./training/{revision}/cldm_v21_v1.yaml"
start_model = f"./training/{revision}/models/{MODEL}"
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
prompt_chance = 0.7
control_chance = 0.8
control_type = CONTROL_TYPE
sd_locked = True
only_mid_control = False


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
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

# Train!
trainer.fit(model, dataloader)
