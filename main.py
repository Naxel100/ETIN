import hydra
from scripts.etin import ETIN
import warnings


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    import time
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print()
        # Create an Expression Tree Improver Network (ETIN)
        etin = ETIN(cfg.Language, cfg.Model)
        # Train the ETIN
        etin.train(cfg.Train)
        # ETIN ready to be used. Example:
        # etin.improve(example_expression_tree)
    
    print('EL MEGATIME:', time.time() - start)

main()
