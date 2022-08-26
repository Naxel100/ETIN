import numpy as np
import sympy as sp



class Token():
    def __init__(self, function, arity, symbol, inv=None, sympy_function=None):
        self.function = function
        self.arity = arity
        self.symbol = symbol
        self.inv = inv
        self.sympy_function = sympy_function


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))

def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    return np.where(np.abs(x1) > 1e-5, np.log(np.abs(x1)), np.zeros(1))

def _protected_pow(x1, x2):
    """Closure of power for negative arguments."""
    return np.power(np.abs(x1), x2)

def _np_cube(x):
    return np.power(x, 3)

def _np_fourth_power(x):
    return np.power(x, 4)

def _subtract(x1, x2):
    return x1 - x2

def _divide(x1, x2):
    return x1 / x2

def _square(x):
    return x ** 2

def _cube(x):
    return x ** 3

def _fourth_power(x):
    return x ** 4

def _inv(x):
    return 1 / x

def _neg(x):
    return -x

class Language():
    # Dicts so we can change fastly between token, idx and symbol -> idx 0 is reserved for constants and 1 to 10 for variables
    symbol_to_idx = None
    symbol_to_token = {
        '+': Token(function=np.add, arity=2, symbol='+', inv='-', sympy_function=sp.Add),
        '-': Token(function=np.subtract, arity=2, symbol='-', inv='+', sympy_function=_subtract),
        '*': Token(function=np.multiply, arity=2, symbol='*', inv='/', sympy_function=sp.Mul),
        '/': Token(function=np.divide, arity=2, symbol='/', inv='*', sympy_function=_divide),
        '^2': Token(function=np.square, arity=1, symbol='^2', inv='sqrt', sympy_function=_square),
        '^3': Token(function=_np_cube, arity=1, symbol='^3', sympy_function=_cube),
        '^4': Token(function=_np_fourth_power, arity=1, symbol='^4', sympy_function=_fourth_power),
        'sin': Token(function=np.sin, arity=1, symbol='sin', sympy_function=sp.sin),
        'cos': Token(function=np.cos, arity=1, symbol='cos', sympy_function=sp.cos),
        'exp': Token(function=np.exp, arity=1, symbol='exp', inv='log', sympy_function=sp.exp),
        'log': Token(function=_protected_log, arity=1, symbol='log', inv='exp', sympy_function=sp.log),
        'sqrt': Token(function=_protected_sqrt, arity=1, symbol='sqrt', inv='^2', sympy_function=sp.sqrt),
        'abs': Token(function=np.abs, arity=1, symbol='abs', sympy_function=sp.Abs),
        'max': Token(function=np.maximum, arity=2, symbol='max', sympy_function=sp.Max),
        'min': Token(function=np.minimum, arity=2, symbol='min', sympy_function=sp.Min),
        'inv': Token(function=np.reciprocal, arity=1, symbol='inv', sympy_function=_inv),
        'neg': Token(function=_neg, arity=1, symbol='neg', sympy_function=_neg),
        '^': Token(function=_protected_pow, arity=2, symbol='^', sympy_function=sp.Pow)
    }

    def __init__(self, cfg):
        self.restrictions_for = get_restrictions(cfg.restrictions)
        # Initialize the dictionaries
        assert cfg.max_variables < 6, 'Max variables can not be over 5'

        # Assert that all the function symbols are valid
        self.function_set_symbols = list(zip(*cfg.operators))[0]  # Get the symbols from the operators
        weights = list(zip(*cfg.operators))[1]  # Get the unnormalized probabilities from the operators
        self.P = np.concatenate((np.array([cfg.prob_terminal/cfg.max_variables for i in range(cfg.max_variables)]), 
                 np.array(weights) * (1 - cfg.prob_terminal) / sum(np.array(weights))))  # Probabilities of each element
        self.const_index = len(self.P) if cfg.use_constants else -1 # Index of the constant
        self.ini_index = len(self.P) + cfg.use_constants # To indicate the start of a new expression
        self.padd_index = len(self.P) + cfg.use_constants + 1 # Pading index

        # Functions used in the language
        self.function_set_symbols = self.function_set_symbols   # List of functions to be used
        self.symbol_to_idx = {s: i + cfg.max_variables for i, s in enumerate(self.function_set_symbols)}
        self.function_set_idx = [self.symbol_to_idx[x] for x in self.function_set_symbols]  # List of idx to be used
        self.function_set = [self.symbol_to_token[function] for function in self.function_set_symbols]  # List of tokens to be used
        self.use_constants = cfg.use_constants   # Boolean to indicate if constants should be used

        # Complete dictionaries
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}
        self.token_to_idx = {}
        for symbol, idx in self.symbol_to_idx.items():
            self.token_to_idx[self.symbol_to_token[symbol]] = idx
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}

        # Maximum number of variables
        self.n_functions = len(self.symbol_to_token)
        self.max_variables = cfg.max_variables
        self.size = self.max_variables + len(self.idx_to_symbol) + self.use_constants
        self.prob_add_const = cfg.prob_add_const
        self.prob_mult_const = cfg.prob_mult_const
        self.max_len = cfg.max_len

        # Extra indices
        self.ini_idx = self.size
        self.padding_idx = self.size + 1

        # Language info for building the model
        self.info_for_model = {
            'language_size': self.size,
            'max_len': self.max_len,
            'max_variables': self.max_variables, 
            'padding_idx': self.padding_idx,
            'ini_idx': self.ini_idx
        }


    def get_possibilities(self, function=None, trigo_offspring=None, exp_offspring=None,
                            distributive=None, needed=None, n_variables=None, to_take='all'):

        not_allowed = []
        try: not_allowed += self.restrictions_for[function]
        except: pass

        if n_variables == 1:
            for symbol, token in self.symbol_to_token.items():
                if token.arity == 2:
                    not_allowed.append(symbol)

        if exp_offspring:
            not_allowed += ['exp', 'pow', '^', '^2', '^3', '^4']

        if trigo_offspring:
            not_allowed += ['sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'arcsin', 'arccos', 'arctan', 'arctanh', 'arccosh', 'arcsinh', 'arccosh', 'arctanh', 'exp']

        if distributive:
            not_allowed += [int(x) for x in distributive]
        if function and needed + self.symbol_to_token[function].arity > n_variables - len(distributive):
            not_allowed += [function]

        terminals = list(range(n_variables))
        if to_take == 'all':
            return list(set(list(self.function_set_symbols) + terminals) - set(not_allowed))
        if to_take == 'operator':
            return list(set(self.function_set_symbols) - set(not_allowed))
        return list(set(terminals) - set(not_allowed))



def get_restrictions(restrictions):
    result = {}
    for restriction in restrictions:
        if restriction == 'no_inverse_parent':
            for symbol, token in Language.symbol_to_token.items():
                if token.inv is not None:
                    result[symbol] = result.get(symbol, []) + [token.inv]

        elif restriction == 'no_sqrt_in_log':
            result['log'] = result.get('log', []) + ['sqrt']
        
        elif restriction == 'no_double_division':
            result['/'] = result.get('/', []) + ['/', 'inv']
            result['inv'] = result.get('inv', []) + ['/', 'inv']
        
        elif restriction == 'no_sqrt_in_sqrt':
            result['sqrt'] = result.get('sqrt', []) + ['sqrt']

        elif 'no_pow_in_pow':
            result['^'] = result.get('^', []) + ['^', '^2', '^3', '^4']
            result['^2'] = result.get('^2', []) + ['^', '^2', '^3', '^4']
            result['^3'] = result.get('^3', []) + ['^', '^2', '^3', '^4']
            result['^4'] = result.get('^4', []) + ['^', '^2', '^3', '^4']

        elif 'only_terminals_after_pow':
            result['^'] = result.get('^', []) + list(Language.symbol_to_token.keys())

        elif 'no_log_in_trigo':
            result['sin'] = result.get('sin', []) + ['log']
            result['cos'] = result.get('cos', []) + ['log']
            result['tan'] = result.get('tan', []) + ['log']

        else:
            raise Exception('Unknown restriction')
        
    return result