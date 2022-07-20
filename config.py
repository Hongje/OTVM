from pickle import FALSE, TRUE
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 8
# Specific random seed, -1 for random.
_C.SYSTEM.RANDOM_SEED = 111
_C.SYSTEM.OUTDIR = 'train_log'
_C.SYSTEM.CUDNN_BENCHMARK = True
_C.SYSTEM.CUDNN_DETERMINISTIC = False
_C.SYSTEM.CUDNN_ENABLED = True
_C.SYSTEM.TESTMODE = False

_C.DATASET = CN()
# dataset path
_C.DATASET.PATH = 'PATH/TO/DATASET'
_C.DATASET.MIN_EDGE_LENGTH = 1088

_C.TEST = CN()
_C.TEST.MEMORY_MAX_NUM = 5 # 2: First&Prev, 0: First, 1: Prev, 3~: Multiple
_C.TEST.MEMORY_SKIP_FRAME = 10

_C.TRAIN = CN()
_C.TRAIN.STAGE = 1
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.BASE_LR = 1e-5
_C.TRAIN.LR_STRATEGY = 'stair' # 'poly', 'const' or 'stair'
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.TRAIN_INPUT_SIZE = (320,320)
_C.TRAIN.FRAME_NUM = 3
_C.TRAIN.FREEZE_BN = True

# optimizer type
_C.TRAIN.OPTIMIZER = 'radam' #adam, radam
_C.TRAIN.TOTAL_EPOCHS = 200
_C.TRAIN.IMAGE_FREQ = -1
_C.TRAIN.SAVE_EVERY_EPOCH = 20

_C.ALPHA = CN()
_C.ALPHA.MODEL = 'fba'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`