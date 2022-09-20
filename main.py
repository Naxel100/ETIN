import hydra
from scripts.etin import ETIN
import warnings


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    import time
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Create an Expression Tree Improver Network (ETIN)
        etin = ETIN(cfg.Language, cfg.Model, seed=42)
        # Train the ETIN
        etin.train(cfg.Train)
        # ETIN ready to be used. Example:
        # etin.improve(example_expression_tree)
    
    print('OVERALL EXECUTION TIME:', time.time() - start)

if __name__ == '__main__':
    main()
