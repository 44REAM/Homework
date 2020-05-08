
class Config():
    BATCHSIZE = 16
    N_TRIALS = 5
    EPOCHS = 10
    N_CLASSES = 1
    NAME = 'test'

    MODEL_DIR = './checkpoints/' + NAME
    DB_NAME = 'sqlite:///' + NAME + '.db'