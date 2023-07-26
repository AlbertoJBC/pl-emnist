from argparse import ArgumentParser

import model as pl_model
import trainer as pl_trainer 
import dataset as pl_dataset

# Global Parameters
INPUT_SIZE = 28*28
NUM_CLASSES = 47
BATCH_SIZE = 50
NUM_WORKERS = 4

# CL script
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=None, required=True)
    parser.add_argument("--accelerator", default='gpu', required=True)
    parser.add_argument("--devices", default=0, required=True)
    args = parser.parse_args()

    # Define Model
    model = pl_model.LightningModel(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
    
    # Define Trainer
    trainer = pl_trainer.create_trainer(args.accelerator, args.devices)

    # Define DataModule
    data_module = pl_dataset.LightningDataModule(args.datadir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Initialize training, validation and testing
    trainer.fit(model, data_module)
    trainer.validate(model, data_module)
    trainer.test(model, data_module)