from lightning.pytorch.callbacks import Callback

class MySimpleCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print('[INFO] Training has begun.')

    def on_train_end(self, trainer, pl_module):
        print('[INFO] Training has ended.')