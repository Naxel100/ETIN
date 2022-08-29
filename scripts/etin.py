from copy import deepcopy
from socketserver import DatagramRequestHandler
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
from .expression import Expression
from pytorch_lightning.strategies.ddp import DDPStrategy
from multiprocessing import Manager
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ETIN():
    def __init__(self, language_cfg, model_cfg, seed=69):
        
        # Seed everything to obtain reproducible results
        self.seed = seed
        seed_everything(seed)
        # Save the model configuration
        self.model_cfg = model_cfg
        # Create Language of the the Expressions with the given restrictions
        self.language = Language(language_cfg)
        # Create an Expression Tree Improver Network (ETIN)
        self.from_ckpt = model_cfg.from_path is not None
        if not self.from_ckpt:
            self.etin_model = ETIN_model(model_cfg, self.language.info_for_model)
        else:
            self.etin_model = ETIN_model.load_from_checkpoint(model_cfg.from_path, cfg=self.model_cfg, info_for_model=self.language.info_for_model)
                               


    def train(self, train_cfg):
        self.etin_model.add_train_cfg(train_cfg.Supervised_Training)

        # Supervised training
        # self.supervised_training(train_cfg.Supervised_Training)

        # Unsupervised training
        self.rl_training(train_cfg.RL_Training)
                
    
    def preprocessing(self, data):
        input_dataset, target_expressions = [], []

        for row in data:
            a, b = create_input(row, self.language)
            input_dataset.append(a)
            target_expressions.append(b)

        input_dataset = pad_sequence(input_dataset, padding_value=0).permute(1, 0, 2)
        target_expressions = pad_sequence(target_expressions, padding_value=self.language.padding_idx).permute(1, 0)

        return (input_dataset, target_expressions)

    

    def supervised_training(self, train_cfg):
        # Add train_cfg to the model
        self.etin_model.add_train_cfg(train_cfg)

        # Create a logger to save the partial results
        logger = CSVLogger("logs", name="logs") 

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="Weights/",                 
            filename="log_{epoch:02d}_{val_loss:.2f}",
            mode="min",
        )

        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min")

        # Create the trainer
        trainer = pl.Trainer(
            # strategy=DDPStrategy(find_unused_parameters=False),
            gpus=train_cfg.gpus,
            max_epochs=train_cfg.epochs,
            logger=logger,
            deterministic=True,
            precision=train_cfg.precision,
            log_every_n_steps=train_cfg.log_frequency,
            callbacks=[checkpoint_callback, early_stop_callback],
            detect_anomaly=True,
            resume_from_checkpoint=self.model_cfg.from_path,
        )

        train_data = Dataset(train_cfg.n_functions_train, self.language)
        train_dataloader = DataLoader(train_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)
        # Test Data
        test_data = Dataset(train_cfg.n_functions_test, self.language)
        test_dataloader = DataLoader(test_data, batch_size=train_cfg.batch_size, collate_fn=self.preprocessing)

        # Fit the model with the train and test data
        trainer.fit(self.etin_model, train_dataloader, test_dataloader)

        # Recover the best model for the next mega epoch
        self.ckpt_path = checkpoint_callback.best_model_path
        self.etin_model = self.etin_model.load_from_checkpoint(self.ckpt_path, cfg=self.model_cfg, info_for_model=self.language.info_for_model)
        self.etin_model.add_train_cfg(train_cfg)



    def rl_training(self, train_cfg, seed_again=True):

        if seed_again:
            seed_everything(self.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.etin_model.to(device)
        optimizer = torch.optim.Adam(self.etin_model.parameters(), lr=train_cfg.lr)
        data = Dataset(train_cfg.episodes, self.language)

        def compute_reward(y_pred, y_true, expression):
            # Compute NORMALIZED RMSE between the prediction and the actual y and add a penalty for complexity of the expression
            std_y_true = np.std(y_true)
            nrmse = np.sqrt(np.mean((y_pred - y_true)**2)) / std_y_true
            complexity = 0  # ESTO CAMBIARLO POR UNA BUENA MEDIDA DE COMPLEJIDAD
            x =  nrmse + complexity * train_cfg.complexity_penalty
            return train_cfg.squishing_scale / (1 + x)
        
        history, control = ResultsContainer(), ResultsContainer()

        initial_discover_probability = train_cfg.discover_probability
        discover_probability = initial_discover_probability

        for episode, row in enumerate(data):
            saved_probs, rewards = [], []

            # Generate an episode
            new_expr = Expression(self.language, model=self.etin_model, prev_info=row, 
                                  record_probabilities=True, discover_probability=discover_probability)
            y_pred = new_expr.evaluate(row['X'])
            if (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).min() < 1e-2):
                continue
            saved_probs += new_expr.probabilities
            expression_reward = compute_reward(y_pred, row['y'], new_expr)
            rewards = [expression_reward for _ in range(len(new_expr.probabilities))]
                

            if rewards:
                control.scores.append(expression_reward)

                rewards = torch.Tensor(rewards).to(device)
                log_probs = torch.log(torch.stack(saved_probs))
                policy_gradient = (-rewards * log_probs).mean()

                # Optimize
                optimizer.zero_grad()
                policy_gradient.backward()
                optimizer.step()   # Mirar porque seguramente aqui hay un fallo

                
                control.log_probs.append(torch.mean(log_probs.detach().cpu()).item())
                control.loss.append(policy_gradient.detach().cpu().numpy())

                del log_probs, rewards, new_expr

            if (episode + 1) % train_cfg.control_each == 0:
                print('Episode', episode + 1, 'score', np.mean(control.scores), 'loss', np.mean(control.loss), 'log_probs', np.mean(control.log_probs))
                history.scores.append(np.mean(control.scores))
                history.loss.append(np.mean(control.loss))
                history.log_probs.append(np.mean(control.log_probs))
                control.reset()
                
            if (episode + 1) % train_cfg.save_each == 0:
                state = {
                    'epoch': episode + 1,
                    'state_dict': self.etin_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scores': history.scores,
                    'loss': history.loss,
                    'log_probs': history.log_probs
                }
                torch.save(state, train_cfg.model_path+'model_'+str(episode + 1)+'.pt')

            discover_probability = initial_discover_probability * train_cfg.decay ** episode


def get_memory():
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f

class ResultsContainer:
    def __init__(self):
        self.scores = []
        self.log_probs = []
        self.loss = []
    
    def reset(self):
        self.__init__