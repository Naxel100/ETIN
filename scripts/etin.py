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
        self.etin_model.add_train_cfg(train_cfg)

        # Supervised training
        self.supervised_training(train_cfg)

        # Unsupervised training
        self.unsupervised_training(train_cfg)
                
    
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

        for mega_epoch in range(self.language.memory_size + 1):  # For each mega epoch

            # Create a logger to save the partial results
            logger = CSVLogger("logs", name="mega_epoch"+str(mega_epoch)) 

            # Create the trainer
            trainer = pl.Trainer(
                gpus=train_cfg.gpus,
                max_epochs=train_cfg.epochs,
                logger=logger,
            )

            # Train Data
            train_data = Dataset(train_cfg.n_functions_train, self.language, max_passes=mega_epoch, model=self.etin_model)
            idx = -1
            for i in range(len(train_data)):
                if train_data[i]['Input Expressions']:
                    idx = i
                    break
            if idx != -1:
                print('----------------------------------------')
                print('OBSERVATION INFORMATION')
                print('----------------------------------------')
                obs = train_data[idx]
                print('n_obs:', obs['n_obs'])
                print('Target Expression:', obs['Target Expression'].traversal)
                print('Constants:', obs['Target Expression'].constants)
                print('X[0]:', obs['X'][idx])
                print('y[0]:', obs['y'][idx])
                print('----------------------------------------\n')
                print('PREVIOUS EXPRESSIONS')
                print('----------------------------------------')
                for i in range(len(obs['Input Expressions'])):
                    print('Input Expression:', obs['Input Expressions'][i].traversal)
                    print('Constants:', obs['Input Expressions'][i].constants)
                print('----------------------------------------')
                a = bbbbb

            train_dataloader = DataLoader(train_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)
            # Test Data
            test_data = Dataset(train_cfg.n_functions_test, self.language, max_passes=mega_epoch, model=self.etin_model)
            test_dataloader = DataLoader(test_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)

            # Fit the model with the train and test data
            trainer.fit(self.etin_model, train_dataloader, test_dataloader)

            # Save the model
            # torch.save(self.etin_model.state_dict(), 'tmp/.x2go-alexfm/media/disk/_cygdrive_C_Users_UX331U_DOCUME1_TFG_Code_ETIN/weights/mega_epoch_'+str(mega_epoch)+'.pt')
