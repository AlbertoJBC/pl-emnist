import torch
import lightning.pytorch as pl

# Callbacks
from lightning.pytorch.callbacks import EarlyStopping 
import callbacks as pl_callbacks

# Logger
from lightning.pytorch.loggers import TensorBoardLogger
tb_logger = TensorBoardLogger(save_dir='tb_logs', name='emnist-0')

# Profiler
from lightning.pytorch.profilers import PyTorchProfiler
profiler = PyTorchProfiler(
    on_trace_ready=torch.profiler.tensorboard_trace_handler('tb_logs/pt-profiler0'),
    schedule=torch.profiler.schedule(skip_first=5, wait=1, warmup=1, active=20)
)

# Trainer definition
def create_trainer(acc, devices):
    trainer = pl.Trainer(fast_dev_run=False,
                        min_epochs=1,
                        max_epochs=1000,
                        accelerator=str(acc),
                        devices=[int(devices)],
                        precision=16, # Default to 32
                        check_val_every_n_epoch=5,
                        callbacks=[
                            EarlyStopping(monitor='val_loss'),
                            pl_callbacks.MySimpleCallback(),
                        ],
                        logger=tb_logger,
                        profiler=profiler)
    return trainer