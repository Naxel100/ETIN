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
        # self.supervised_training(train_cfg.Supervised_Training)

        # Unsupervised training
        self.rl_training(train_cfg.RL_Training)
                
    
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



    def rl_training(self, train_cfg):

        # torch.nn.utils.clip_grad_value_(self.etin_model.parameters(), clip_value=1)

        optimizer = torch.optim.Adam(self.etin_model.parameters(), lr=train_cfg.lr)

        def compute_reward(y_pred, y_true, expression):
            # Compute NORMALIZED RMSE between the prediction and the actual y and add a penalty for complexity of the expression
            std_y_true = np.std(y_true)
            nrmse = np.sqrt(np.mean((y_pred - y_true)**2)) / std_y_true
            complexity = 0  # ESTO CAMBIARLO POR UNA BUENA MEDIDA DE COMPLEJIDAD
            x =  nrmse + complexity * train_cfg.complexity_penalty
            return train_cfg.squishing_scale / (1 + x)


        data = Dataset(train_cfg.episodes, self.language)
        
        scores_control = []
        gradients_control = []
        history_scores = []
        history_gradients = []

        for episode, row in enumerate(data):
            saved_probs = []
            rewards = []
            
            # Generate an episode
            for t in range(train_cfg.max_expressions):
                new_expr = Expression(self.language, model=self.etin_model, prev_info=row, record_probabilities=True)
                # Update prev_info
                y_pred = new_expr.evaluate(row['X'])
                if (np.isnan(y_pred).any() or np.abs(y_pred).max() > 1e5 or np.abs(y_pred).min() < 1e-2):
                    break
                saved_probs += new_expr.probabilities
                rewards += [0 for _ in range(len(new_expr.probabilities) - 1)] + [compute_reward(y_pred, row['y'], new_expr)]
                if len(row['Input Expressions']) > self.language.memory_size:
                    row['Input Expressions'][t % self.language.memory_size] = new_expr
                    row['y_preds'][t % self.language.memory_size] = y_pred
                else:
                    row['Input Expressions'].append(new_expr)
                    row['y_preds'].append(y_pred)

            if rewards:
                scores_control.append(np.sum(rewards))
                
                # Compute the gradient through REINFORCE algorithm
                discounted_rewards = []
                for t in range(len(rewards)):
                    Gt = 0
                    pw = 0
                    for r_ in rewards[t:]:
                        Gt += r_ * train_cfg.gamma ** pw
                        pw += 1
                    discounted_rewards.append(Gt)

                discounted_rewards = torch.Tensor(discounted_rewards)
                # Ver si tiene sentido normalizar los discounted rewards o no
                # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
                log_probs = torch.log(torch.stack(saved_probs))
                policy_gradient = (-discounted_rewards * log_probs).sum()

                # Optimize
                optimizer.zero_grad()
                policy_gradient.backward()
                optimizer.step()   # Mirar porque seguramente aqui hay un fallo

                gradients_control.append(policy_gradient.detach().numpy())

            if (episode + 1) % train_cfg.control_each == 0:
                print('Episode', episode + 1, 'score', np.mean(scores_control), 'loss', np.mean(gradients_control))
                history_scores.append(np.mean(scores_control))
                history_gradients.append(np.mean(gradients_control))
                scores_control = []
                
            if (episode + 1) % train_cfg.save_each == 0:
                state = {
                    'epoch': episode + 1,
                    'state_dict': self.etin_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scores': history_scores,
                    'gradients': history_gradients,
                }
                torch.save(state, 'C:/Users/UX331U/Documents/TFG/Code/ETIN/outputs/rl/model_'+str(episode + 1)+'.pt')


