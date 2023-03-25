import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class TB_Dataset(Dataset):
    def __init__(self, control_type, revision="r1", prompt_chance=0.8, control_chance=0.8):
        self.control_chance = control_chance
        self.prompt_chance = prompt_chance
        self.control_type = control_type
        self.revision_folder=f"./training/{revision}/"
        self.train_db = []
        with open(self.revision_folder+'train_db.json', 'r') as f:
            self.train_db = json.load(f)
        print(f"-----------Loaded {len(self.train_db)} entries from training DB--------")
        print(f"-----------Targeting {self.control_type}----------")

    def __len__(self):
        return len(self.train_db)

    def __getitem__(self, idx):
        item = self.train_db[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(self.revision_folder + f"{self.control_type}/"+source_filename)
        target = cv2.imread(self.revision_folder + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

