import json
import cv2
import numpy as np

import random
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from distortions import random_distortion

from config import *


class TB_Dataset(Dataset):
    def __init__(self, control_type, revision="r1", prompt_chance=0.8, control_chance=0.8):
        self.control_chance = control_chance
        self.prompt_chance = prompt_chance
        self.control_type = control_type
        self.train_db = []
        with open(TRAINDB_LOCAL, 'r') as f:
            self.train_db = json.load(f)
        print(f"-----------Loaded {len(self.train_db)} entries from training DB--------")
        print(f"-----------Targeting {self.control_type}----------")

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, idx):
        item = self.train_db[idx]

        source_filename = item['source'].split("/")[-1] #the control input
        target_filename = item['target'].split("/")[-1] #the real image
        prompt = item['prompt']

        source = cv2.imread(CONTROLS_EXTRACT + source_filename)
        target = cv2.imread(IMAGES_EXTRACT + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        #rescale source
        source = source - (np.min(source))
        source = source * (255/(np.max(source)))

        if (random.uniform(0.0, 1.0) > self.prompt_chance): 
            prompt = " " #delete the prompt
        elif (random.uniform(0.0, 1.0) > self.control_chance):
            source = np.zeros_like(source, dtype=np.uint8) #delete the control

        #source = random_distortion(source)

        # Normalize source images to [-1, 1].
        source = (source.astype(np.float32) / 127.5)-1.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class TB_Sampler(Sampler):
    def __init__(self, dataset, subset_size):
        self.dataset = dataset
        self.subset_size = subset_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        subset_indices = random.sample(indices, self.subset_size)
        return iter(subset_indices)

    def __len__(self):
        return self.subset_size

if __name__=='__main__':
    print("testing TB_Dataset...")
    tb=TB_Dataset('sketch',revision='r1')
    for x in range(0,10000):
        t=tb[x]

