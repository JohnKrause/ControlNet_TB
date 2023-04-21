save_memory = False

CONTROL_TYPE='sketch'
REVNUM="r1"

CONTROLS_TARGET=f"controls_{CONTROL_TYPE}_{REVNUM}.tar.xz"
CONTROLS_TARGET_LOCAL=f'{CONTROLS_TARGET}'
CONTROLS_EXTRACT=f'training/{REVNUM}/{CONTROL_TYPE}/controls/'

IMAGES_TARGET=f"images_resize_{REVNUM}.tar.xz"
IMAGES_TARGET_LOCAL=f'{IMAGES_TARGET}'
IMAGES_EXTRACT=f'training/{REVNUM}/images_resize/'

MODEL="sd21_control_v1.ckpt"
MODEL_LOCAL=f'training/{REVNUM}/models/{MODEL}'

TRAINDB=f"train_db_{REVNUM}.json"
TRAINDB_LOCAL=f'training/{REVNUM}/{TRAINDB.replace(f"_{REVNUM}","")}'

MODEL_CONFIG_LOCAL=f"training/{REVNUM}/cldm_v21_v1.yaml"
 
LOG_CKPT_PATH=f"{REVNUM}/{CONTROL_TYPE}/"

LOG_IMG_PATH=f"{REVNUM}/{CONTROL_TYPE}/examples/"
