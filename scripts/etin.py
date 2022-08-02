from .language import Language
from .dclasses import Dataset
from torch.utils.data import DataLoader
import torch
from .model import (
    ETIN_model,
    create_input
)
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.loggers import CSVLogger
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import time


class ETIN():
    def __init__(self, language_cfg, model_cfg, seed=69):
        
        # Seed everything to obtain reproducible results
        seed_everything(seed)
        # Save the model configuration
        self.model_cfg = model_cfg
        # Create Language of the the Expressions with the given restrictions
        self.language = Language(language_cfg)
        # Create an Expression Tree Improver Network (ETIN)
        self.etin_model = ETIN_model(model_cfg, self.language.info_for_model)                           


    def train(self, train_cfg):
        self.etin_model.add_train_cfg(train_cfg.Supervised_Training)

        # Supervised training
        self.supervised_training(train_cfg.Supervised_Training)

        # Unsupervised training
        # self.unsupervised_training(train_cfg.Unsupervised_Training)
                
    
    def preprocessing(self, data):
        input_dataset, target_expressions, input_expressions = [], [], []

        for row in data:
            a, b, c = create_input(row, self.language)
            input_dataset.append(a)
            input_expressions.append(b)
            target_expressions.append(c)

        for_loss = torch.Tensor([(len(expr) - 1) % (self.language.max_len + 1) for expr in input_expressions])

        input_dataset = pad_sequence(input_dataset, padding_value=0).permute(1, 0, 2)
        input_expressions = pad_sequence(input_expressions, padding_value=self.language.padding_idx).permute(1, 0)
        target_expressions = pad_sequence(target_expressions, padding_value=self.language.padding_idx).permute(1, 0)

        return (input_dataset, input_expressions, target_expressions, for_loss)

    

    def supervised_training(self, train_cfg):
        # Add train_cfg to the model
        self.etin_model.add_train_cfg(train_cfg)

        mega_epochs = self.language.memory_size + 1
        assert (len(train_cfg['n_functions_train']) == mega_epochs), "The number of functions to train must be equal to the number of mega epochs"
        assert (len(train_cfg['n_functions_test'])  == mega_epochs), "The number of functions to test must be equal to the number of mega epochs"

        for mega_epoch in range(mega_epochs):  # For each mega epoch

            print('\n\n------------------------------------------------------------------------------------------------------------')
            print('\nMega epoch:', mega_epoch, '\n')
            
            start_time = time.time()

            # Create a logger to save the partial results
            logger = CSVLogger("logs", name="mega_epoch"+str(mega_epoch)) 

            # Create the trainer
            trainer = pl.Trainer(
                gpus=train_cfg.gpus,
                max_epochs=train_cfg.epochs,
                logger=logger,
                deterministic=True,
                precision=train_cfg.precision,
                log_every_n_steps=train_cfg.log_frequency,
            )

            # Train Data
            train_data = Dataset(train_cfg.n_functions_train[mega_epoch], self.language, max_passes=mega_epoch, model=self.etin_model)
            train_dataloader = DataLoader(train_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)
            # Test Data
            test_data = Dataset(train_cfg.n_functions_test[mega_epoch], self.language, max_passes=mega_epoch, model=self.etin_model)
            test_dataloader = DataLoader(test_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)

            # Fit the model with the train and test data
            trainer.fit(self.etin_model, train_dataloader, test_dataloader)

            print('Mega Epoch', mega_epoch, 'took', time.time() - start_time, 'seconds.\n')
            print('------------------------------------------------------------------------------------------------------------\n')

            # Save the model
            # torch.save(self.etin_model.state_dict(), 'tmp/.x2go-alexfm/media/disk/_cygdrive_C_Users_UX331U_DOCUME1_TFG_Code_ETIN/weights/mega_epoch_'+str(mega_epoch)+'.pt')
