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
import warnings
from .expression import Expression
from pytorch_lightning.strategies.ddp import DDPStrategy
from multiprocessing import Manager
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def nrmse(y_pred, y_true):
    std_y_true = np.std(y_true)
    return np.sqrt(np.mean((y_pred - y_true)**2)) / std_y_true


class ETIN():
    def __init__(self, language_cfg, model_cfg, seed=69, from_path=None):
        
        # Seed everything to obtain reproducible results
        seed_everything(seed)
        # Save the model configuration
        self.model_cfg = model_cfg
        # Create Language of the the Expressions with the given restrictions
        self.language = Language(language_cfg)
        # Create an Expression Tree Improver Network (ETIN)
        if from_path is None and model_cfg.from_path is not None:
            from_path = model_cfg.from_path
        self.from_ckpt = from_path is not None
        if not self.from_ckpt:
            self.etin_model = ETIN_model(model_cfg, self.language.info_for_model)
        else:
            self.etin_model = ETIN_model.load_from_checkpoint(model_cfg.from_path, cfg=self.model_cfg, info_for_model=self.language.info_for_model)
                               

    def get_expression(self, X, y, method='random', 
                       max_expressions=100, n_beam=2, n_gens_per_beam=10, beam_mode='mean', 
                       error_function=nrmse):
        
        if method == 'random':
            return self.get_expression_by_random(X, y, max_expressions, error_function)
        
        elif method == 'beam search':
            return self.get_expression_by_bsearch(X, y, n_beam, n_gens_per_beam, error_function, beam_mode)

    
    def get_expression_by_bsearch(self, X, y, n_beam, n_gens_per_beam, error_function, beam_mode):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.etin_model.to(device)

        prev_info = {'X': X, 'y': y}
        input_info, _ = create_input(prev_info, self.language)
        Xy = input_info.unsqueeze(0).to(device)
        enc_src = self.etin_model(Xy, None, only_encoder=True)

        def is_finished(program):
            arity = 1
            for symbol in program:
                if isinstance(symbol, str):
                    arity += (self.language.symbol_to_token[symbol].arity - 1)
                else:
                    arity -= 1
            return arity == 0

        res = []
        program = []
        finished = False


        while not finished:
            best_program = []
            best_score = np.inf
            for i in range(n_beam):
                error_min, error_mean, cont = np.inf, 0, 0
                for k in range(n_gens_per_beam):
                    new_expr = Expression(self.language, model=self.etin_model, 
                                          prev_info=prev_info, enc_src=enc_src, 
                                          partial_expression=program, max_pos=i)
                    y_pred = new_expr.evaluate(X)
                    if (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).max() - np.abs(y_pred).min() == 0):
                        continue
                    error = error_function(y_pred, y)
                    res.append({'expression': new_expr, 'error': error})
                    print(i, k, new_expr.traversal, error)
                    if error < 1e-10:
                        return res
                    error_mean += error
                    cont += 1
                    if error < error_min:
                        error_min = error
                if cont > 0:
                    error_mean /= cont

                if beam_mode == 'mean' and error_mean < best_score:
                    best_score = error_mean
                    new_token = new_expr.traversal[len(program)]
                    new_token = new_token if new_token not in self.language.idx_to_symbol else self.language.idx_to_symbol[new_token]
                    best_program = program + [new_token]

                elif beam_mode == 'min' and error_min < best_score:
                    best_score = error_min
                    new_token = new_expr.traversal[len(program)]
                    new_token = new_token if new_token not in self.language.idx_to_symbol else self.language.idx_to_symbol[new_token]
                    best_program = program + [new_token]
            program = best_program
            print(program)
            finished = is_finished(program)

        return res



    def get_expression_by_random(self, X, y, max_expressions, error_function):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.etin_model.to(device)

        prev_info = {'X': X, 'y': y}
        input_info, _ = create_input(prev_info, self.language)
        Xy = input_info.unsqueeze(0).to(device)
        enc_src = self.etin_model(Xy, None, only_encoder=True)

        res = []

        for _ in range(max_expressions):
            expr = Expression(self.language, model=self.etin_model, prev_info=prev_info, enc_src=enc_src)
            y_pred = expr.evaluate(X)

            if (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).max() - np.abs(y_pred).min() == 0):
                continue
            error = error_function(y_pred, y)
            res.append({'expression': expr, 'error': error})
        return res
            

    def train(self, train_cfg):
        self.etin_model.add_train_cfg(train_cfg.Supervised_Training)

        # Supervised training
        self.supervised_training(train_cfg.Supervised_Training)

        # Unsupervised training
        # self.rl_training(train_cfg.RL_Training)
                
    
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

        if train_cfg.seed is not None:
            seed_everything(train_cfg.seed)

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
        # self.ckpt_path = checkpoint_callback.best_model_path
        # self.etin_model = self.etin_model.load_from_checkpoint(self.ckpt_path, cfg=self.model_cfg, info_for_model=self.language.info_for_model)
        # self.etin_model.add_train_cfg(train_cfg)



    def rl_training(self, train_cfg):
        if train_cfg.seed is not None:
            seed_everything(train_cfg.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.etin_model.to(device)
        optimizer = torch.optim.Adam(self.etin_model.parameters(), lr=train_cfg.lr)
        data = Dataset(train_cfg.episodes, self.language)
        
        def compute_reward(y_pred, y_true, expression):
            # Compute NORMALIZED RMSE between the prediction and the actual y and add a penalty for complexity of the expression
            nrmse_res = nrmse(y_pred, y_true)
            complexity = 0
            for token in expression.traversal:
                if token in self.language.idx_to_symbol:
                    complexity += self.language.complexity[self.language.idx_to_symbol[token]]
            return train_cfg.squishing_scale / (1 + nrmse_res) - train_cfg.complexity_penalty * complexity, nrmse_res
        
        history, control = ResultsContainer(), ResultsContainer()
        initial_discover_probability = train_cfg.discover_probability

        for episode, row in enumerate(data):
            saved_probs, rewards, max_rewards, nrmses = [], [], [], []
            max_reward = -1
            discover_probability = initial_discover_probability * train_cfg.decay**episode
            if train_cfg.num_expressions > 1:
                input_info, _ = create_input(row, self.language)
                Xy = input_info.unsqueeze(0).to(device)
                enc_src = self.etin_model(Xy, None, only_encoder=True)
            else:
                enc_src = None

            for _ in range(train_cfg.num_expressions):
                # Generate an episode
                new_expr = Expression(self.language, model=self.etin_model, prev_info=row, enc_src=enc_src,
                                      record_probabilities=True, discover_probability=discover_probability)
                y_pred = new_expr.evaluate(row['X'])
                if (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).max() - np.abs(y_pred).min() == 0):
                    continue

                expression_reward, nrmse_res = compute_reward(y_pred, row['y'], new_expr)
                if expression_reward > max_reward:
                    max_reward = expression_reward
                    saved_probs = new_expr.probabilities
                rewards.append(expression_reward)
                nrmses.append(nrmse_res)
                
            if rewards and saved_probs:
                control.scores.append(np.mean(rewards))
                control.nrmses.append(np.mean(nrmses))
                control.max_scores.append(max_reward)
                log_probs = torch.log(torch.stack(saved_probs))
                episode_rewards = [max_reward for _ in range(len(saved_probs))]
                episode_rewards = torch.Tensor(episode_rewards).to(device)
                policy_gradient = (-episode_rewards * log_probs).mean()

                # Optimize
                optimizer.zero_grad()
                policy_gradient.backward()
                optimizer.step()
                
                control.log_probs.append(torch.mean(log_probs.detach().cpu()).item())
                control.loss.append(policy_gradient.detach().cpu().numpy())

                del log_probs, rewards, new_expr

            if (episode + 1) % train_cfg.control_each == 0:
                # print('Episode', episode + 1, 'max_score', np.mean(control.max_scores), 'score', np.mean(control.scores), 'nrmse', np.mean(control.nrmses), 'loss', np.mean(control.loss), 'log_probs', np.mean(control.log_probs))
                history.max_scores.append(np.mean(control.max_scores))
                history.std_max_scores.append(np.std(control.max_scores))

                history.scores.append(np.mean(control.scores))
                history.std_scores.append(np.std(control.scores))

                history.loss.append(np.mean(control.loss))
                history.std_loss.append(np.std(control.loss))

                history.log_probs.append(np.mean(control.log_probs))
                history.std_log_probs.append(np.std(control.log_probs))

                history.nrmses.append(np.mean(control.nrmses))
                history.std_nrmses.append(np.std(control.nrmses))
                control.reset()
                
            if (episode + 1) % train_cfg.save_each == 0:
                state = {
                    'epoch': episode + 1,
                    'state_dict': self.etin_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'max_scores': history.max_scores,
                    'std_max_scores': history.std_max_scores,
                    'scores': history.scores,
                    'std_scores': history.std_scores,
                    'loss': history.loss,
                    'std_loss': history.std_loss,
                    'log_probs': history.log_probs,
                    'std_log_probs': history.std_log_probs,
                    'nrmses': history.nrmses,
                    'std_nrmses': history.std_nrmses
                }
                torch.save(state, train_cfg.model_path+'/model_'+str(episode + 1)+'.pt')


class ResultsContainer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.max_scores = []
        self.std_max_scores = []
        self.scores = []
        self.std_scores = []
        self.log_probs = []
        self.std_log_probs = []
        self.loss = []
        self.std_loss = []
        self.nrmses = []
        self.std_nrmses = []