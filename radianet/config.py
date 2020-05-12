
class Config():
    BATCHSIZE = 2
    N_TRIALS = 1
    EPOCHS = 2
    N_CLASSES = 1
    BORDER_MODE = 0
    NAME = 'test_chechpoint'
    IMAGE_SIZE = 240
    LIDC_PATH = "D:\\GoogleDrive\\dataset\\radiology\\TCIA_LIDC-IDRI\\preprocess\\"
    EFFICIENTNET_B0_LAYER = 15

    MODEL_DIR = './checkpoints/' + NAME
    DB_NAME = 'sqlite:///' + NAME + '.db'
