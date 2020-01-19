class Config(object):
    def __init__(self):
        #data
        self.TRAINGSET_DIR = "data/train/"
        self.TRAINSET_CSV = "data/training_frames_keypoints.csv"
        self.TESTSET_DIR = "data/test/"
        self.TESTSET_CSV = "data/test_frames_keypoints.csv"
        #model
        self.MODEL_DIR = "runs/"
        self.device = "cuda"
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 15