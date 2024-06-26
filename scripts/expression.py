from cmath import isnan
import numpy as np
from sympy import symbols
from scipy.optimize import minimize
from .model import create_input
import torch
from scipy.stats import entropy


class Expression():
    '''
    This class represents an Expression Tree. It is composed by a list of tokens/traversal (ex: [0, 2, 1]) and its one-hot-encoding representation in torch ([[1,0,0],[0,0,1],[0,1,0]]).
    '''
    def __init__(self, language, traversal=None, record_probabilities=False, enc_src=None, 
                 partial_expression=None, put_max_after_partial=True, max_pos=0,
                 model=None, prev_info=None, constants=None, discover_probability=1.):
        # Save in self the variables needed
        self.language = language
        self.n_variables = language.max_variables

        if traversal is None:
            if model is not None:
                assert prev_info is not None
                input_info, _ = create_input(prev_info, self.language)
                traversal = self.generate_expression_from_model(model, input_info, 
                                                                record_probabilities=record_probabilities,
                                                                discover_probability=discover_probability,
                                                                enc_src=enc_src, partial_expression=partial_expression,
                                                                put_max_after_partial=True, max_pos=max_pos)
            else:
                traversal = self.generate_random_expression(partial_expression=partial_expression)
        
        traversal = [self.language.symbol_to_idx[x] if isinstance(x, str) else x for x in traversal]
        self.traversal = traversal

        if language.use_constants:
            if constants is None:
                self.constants = np.abs(np.random.normal(0, 3, size=(self.traversal.count(language.const_index),)))
                if prev_info is not None:
                    self.constants = self.optimize_constants(prev_info['X'], prev_info['y'])

    def to_sympy(self):
        '''
        This function returns the sympy expression of the expression tree.
        '''
        x1, x2, x3, x4, x5 = symbols('x1, x2, x3, x4, x5')
        x = x1, x2, x3, x4, x5
        stack = []
        const = 0
        for ele in self.traversal[::-1]:
            if ele == self.language.const_index:
                stack.append(self.constants[const])
                const += 1
                
            elif ele in self.language.function_set_symbols or ele >= self.language.max_variables:
                function = self.language.symbol_to_token[ele] if isinstance(ele, str) else self.language.idx_to_token[ele]
                first_operand = stack.pop()
                if function.arity == 1:
                    stack.append(function.sympy_function(first_operand))
                else:
                    second_operand = stack.pop()
                    stack.append(function.sympy_function(first_operand, second_operand))
                
            else:
                stack.append(x[int(ele)])

        return stack[0].simplify()


    def evaluate(self, X, constants=None):
        '''
        This function evaluates the expression tree at a given point x.
        '''
        if constants is None:
            constants = self.constants

        stack = []
        const = 0
        for ele in self.traversal[::-1]:
            if ele == self.language.const_index:
                stack.append(constants[const]*np.ones(X.shape[0]))
                const += 1

            elif ele in self.language.function_set_symbols or ele >= self.language.max_variables:
                function = self.language.symbol_to_token[ele] if isinstance(ele, str) else self.language.idx_to_token[ele]
                first_operand = stack.pop()
                if function.arity == 1:
                    stack.append(function.function(first_operand))
                else:
                    second_operand = stack.pop()
                    stack.append(function.function(first_operand, second_operand))

            else:
                stack.append(X[:, int(ele)])
        
        return stack[0]

    
    def insert_constants(self, choice, hollows=100000):
        to_append = [choice] 
        if np.random.random() < self.language.prob_mult_const and hollows >= 2:
            to_append = ['*'] + [self.language.const_index] + to_append
            hollows -= 2
        if np.random.random() < self.language.prob_add_const and hollows >= 2:
            to_append = ['+'] + [self.language.const_index] + to_append
        return to_append


    
    def add_node_model(self, arities_stack, function_stack, program, will_be_nodes, might_use_constant, P, discover_probability=1., choice=None, max_val=-1):
        if choice is None:
            if not function_stack:
                possibilities = list(self.language.function_set_symbols) + list(range(self.n_variables))
            else:
                possibilities = self.language.get_possibilities(**function_stack[-1], n_variables=self.n_variables)
            
            if self.language.use_constants and might_use_constant:
                possibilities.append(self.language.const_index)

            # Get their correspondent probabilities and normalize them
            indices = sorted([self.language.symbol_to_idx[x] if isinstance(x, str) else x for x in possibilities])
            possibilities = [self.language.idx_to_symbol.get(x, x) for x in indices]
            prob_possibilities = list(map(P.__getitem__, indices))
            prob_possibilities = [p / sum(prob_possibilities) for p in prob_possibilities]
            # Choose a token idx
            if max_val == -1:
                if np.random.random() < discover_probability:
                    choice = np.random.choice(possibilities, p=prob_possibilities)
                else:
                    choice = possibilities[np.argmax(prob_possibilities)]
            else:
                choice = sorted(zip(prob_possibilities, possibilities))[::-1][max_val][1]

        # Determine if we are adding a function or terminal -> Add Function if we got a function and there will be enough nodes to add it and add its children
        if (choice in self.language.function_set_symbols and 
           will_be_nodes + self.language.symbol_to_token[choice].arity <= self.language.max_len):

            to_append = [choice]
            will_be_nodes += self.language.symbol_to_token[choice].arity

            program += to_append  # Append to program
            arities_stack.append(self.language.symbol_to_token[choice].arity)   # Append to arities
            new_set = set()   # If we are adding a function of the same type, we need to inherit its not_allowed variables
            trigo_offspring = False if not function_stack else function_stack[-1]['trigo_offspring']
            exp_offspring = False if not function_stack else function_stack[-1]['exp_offspring']
            needed = self.language.symbol_to_token[choice].arity - 1
            if function_stack and (function_stack[-1]['function'] == choice or
                                   self.language.symbol_to_token[choice].inv == choice):  # If same kind of operand
                might_use_constant = False
                new_set = function_stack[-1]['distributive']
                needed += function_stack[-1]['needed']
            else: might_use_constant = True
            if choice in ['sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'arcsin', 'arccos', 'arctan', 'arctanh', 'arccosh', 'arcsinh', 'arccosh', 'arctanh']:
                trigo_offspring = True
            if choice in ['exp', 'pow', 'pow2', 'pow3', 'pow4', 'pow5']:
                exp_offspring = True
            function_stack.append({'function': choice, 'trigo_offspring': trigo_offspring, 'exp_offspring': exp_offspring, 
                                   'distributive': new_set, 'needed': needed})  # Append to function stack

        else:
            # We need a terminal, add a variable or constant
            if choice in self.language.function_set_symbols:  # If we got here because of the lack of space, we need to choose a terminal again
                # Get the possible terminals and choose one in the same way that it was done before
                possibilities = sorted(self.language.get_possibilities(**function_stack[-1], to_take='terminal', n_variables=self.n_variables))
                if self.language.use_constants and might_use_constant:
                    possibilities.append(self.language.const_index)
                prob_possibilities = list(map(P.__getitem__, possibilities))
                prob_possibilities = [p / sum(prob_possibilities) for p in prob_possibilities]
                if max_val == -1:
                    if np.random.random() < discover_probability:
                        choice = np.random.choice(possibilities, p=prob_possibilities)
                    else:
                        choice = possibilities[np.argmax(prob_possibilities)]
                else:
                    choice = sorted(zip(prob_possibilities, possibilities))[::-1][max_val][1]

            choice = int(choice)
            to_append = [choice]
            program += to_append  # Append to program
            if choice == self.language.const_index:
                might_use_constant = False
            
            if function_stack:
                function_stack[-1]['distributive'].add(choice)  # Add to the not_allowed set
                arities_stack[-1] -= 1  # One node added to the parent, so we need to remove one from the arity
                while arities_stack[-1] == 0:  # If completed arity of node 
                    arities_stack.pop()        # Remove it from the arity stack
                    if not arities_stack:      # If there are no more arities, we are done
                        break
                    child_info = function_stack.pop()  # Remove it from the function stack
                    child_symbol = child_info['function']
                    child_set = child_info['distributive']  # Pass info of non-allowed variables to parent if DistributiveRestriction is used
                    arities_stack[-1] -= 1
                    if child_symbol == function_stack[-1]['function']:
                        function_stack[-1]['distributive'].update(child_set)

        return arities_stack, function_stack, program, will_be_nodes, might_use_constant



    def add_node_random(self, arities_stack, function_stack, program, will_be_nodes, First, P, choice=None):
        if choice is None:
            if First and np.random.random() > self.language.p_degenerated_program:   # If it is the root
                possibilities = self.language.get_possibilities(n_variables=self.n_variables, to_take='operator') # Take a function to avoid degenerated expressions
            else:
                # First let's check what we can add given the context (parent and its not_allowed operands/operators)
                if function_stack:
                    possibilities = self.language.get_possibilities(**function_stack[-1], n_variables=self.n_variables)
                else:
                    possibilities = list(self.language.function_set_symbols) + list(range(self.n_variables))
            # Get their correspondent probabilities and normalize them
            indices = sorted([self.language.symbol_to_idx[x] if isinstance(x, str) else x for x in possibilities])
            possibilities = [self.language.idx_to_symbol.get(x, x) for x in indices]
            prob_possibilities = list(map(P.__getitem__, indices))
            prob_possibilities = [p / sum(prob_possibilities) for p in prob_possibilities]
            # Choose a token idx
            choice = np.random.choice(possibilities, p=prob_possibilities)

        # Determine if we are adding a function or terminal -> Add Function if we got a function and there will be enough nodes to add it and add its children
        if (choice in self.language.function_set_symbols and 
           will_be_nodes + self.language.symbol_to_token[choice].arity <= self.language.max_len):

            to_append = [choice]
            will_be_nodes += self.language.symbol_to_token[choice].arity
            if self.language.use_constants and self.language.symbol_to_token[choice].arity == 1: 
                to_append = self.insert_constants(choice, hollows=self.language.max_len - will_be_nodes)
            will_be_nodes += len(to_append) - 1

            program += to_append  # Append to program
            arities_stack.append(self.language.symbol_to_token[choice].arity)   # Append to arities
            new_set = set()   # If we are adding a function of the same type, we need to inherit its not_allowed variables
            trigo_offspring = False if First else function_stack[-1]['trigo_offspring']
            exp_offspring = False if First else function_stack[-1]['exp_offspring']
            needed = self.language.symbol_to_token[choice].arity - 1
            if not First and (function_stack[-1]['function'] == choice or
                              self.language.symbol_to_token[choice].inv == choice):  # If same kind of operand
                new_set = function_stack[-1]['distributive']
                needed += function_stack[-1]['needed']
            if choice in ['sin', 'cos', 'tan', 'tanh', 'sinh', 'cosh', 'arcsin', 'arccos', 'arctan', 'arctanh', 'arccosh', 'arcsinh', 'arccosh', 'arctanh']:
                trigo_offspring = True
            if choice in ['exp', 'pow', 'pow2', 'pow3', 'pow4', 'pow5']:
                exp_offspring = True
            function_stack.append({'function': choice, 'trigo_offspring': trigo_offspring, 'exp_offspring': exp_offspring, 
                                   'distributive': new_set, 'needed': needed})  # Append to function stack
        else:
            # We need a terminal, add a variable or constant
            if choice in self.language.function_set_symbols:  # If we got here because of the lack of space, we need to choose a terminal again
                # Get the possible terminals and choose one in the same way that it was done before
                possibilities = sorted(self.language.get_possibilities(**function_stack[-1], to_take='terminal', n_variables=self.n_variables))
                prob_possibilities = list(map(P.__getitem__, possibilities))
                prob_possibilities = [p / sum(prob_possibilities) for p in prob_possibilities]
                choice = np.random.choice(possibilities, p=prob_possibilities)

            choice = int(choice)
            to_append = [choice]
            if self.language.use_constants:  # If same kind of operand
                to_append = self.insert_constants(choice, hollows=self.language.max_len - will_be_nodes)
            will_be_nodes += len(to_append) - 1
            program += to_append  # Append to program

            if function_stack:
                function_stack[-1]['distributive'].add(choice)  # Add to the not_allowed set
                arities_stack[-1] -= 1  # One node added to the parent, so we need to remove one from the arity
                while arities_stack[-1] == 0:  # If completed arity of node 
                    arities_stack.pop()        # Remove it from the arity stack
                    if not arities_stack:      # If there are no more arities, we are done
                        break
                    child_info = function_stack.pop()  # Remove it from the function stack
                    child_symbol = child_info['function']
                    child_set = child_info['distributive']  # Pass info of non-allowed variables to parent if DistributiveRestriction is used
                    arities_stack[-1] -= 1
                    if child_symbol == function_stack[-1]['function']:
                        function_stack[-1]['distributive'].update(child_set)

        return arities_stack, function_stack, program, will_be_nodes


    def generate_expression_from_model(self, model, input_info, enc_src=None, partial_expression=None,
                                       record_probabilities=False, device='cuda', discover_probability=1.,
                                       put_max_after_partial=True, max_pos=-1):
        First, might_use_constant = True, True
        will_be_nodes = 1
        arities_stack, function_stack, program, probabilities, entropies = [], [], [], [], []

        # Set to train or eval dpending if we want to record probabilities (for rl part) or not
        if record_probabilities:
            model.train()
        else:
            model.eval()

        Xy = input_info.unsqueeze(0).to(device)
        prev_exprs = None

        while First or arities_stack:   # While there are operands/operators to add
            if partial_expression is not None and len(program) < len(partial_expression):
                choice = partial_expression[len(program)]
                P = None
            else:
                P_original = model(Xy, prev_exprs, enc_src=enc_src).squeeze(0)
                P = P_original.detach().cpu().numpy()[-1]
                choice = None
            max_val = -1
            if put_max_after_partial and partial_expression is not None and len(program) == len(partial_expression):
                max_val = max_pos
            arities_stack, function_stack, program, will_be_nodes, might_use_constant = self.add_node_model(arities_stack, function_stack, program, will_be_nodes, might_use_constant, P, choice=choice, max_val=max_val)
            First = False
            token_idx = self.language.symbol_to_idx[program[-1]] if isinstance(program[-1], str) else program[-1]
            to_append = torch.Tensor([token_idx]).unsqueeze(0).to(device)
            prev_exprs = torch.cat((prev_exprs, to_append), dim=1) if prev_exprs is not None else to_append

            if record_probabilities:
                entropies.append(-torch.sum(P_original[-1] * torch.log(P_original[-1])))
                probabilities.append(P_original[-1, token_idx])
                self.probabilities = probabilities
                self.entropies = entropies
        
        del Xy, prev_exprs

        return program



    def generate_random_expression(self, partial_expression=None):
        First = True
        will_be_nodes = 1
        arities_stack, function_stack, program = [], [], []

        while First or arities_stack:   # While there are operands/operators to add
            choice = None
            if partial_expression is not None and len(program) < len(partial_expression):
                choice = partial_expression[len(program)]
            arities_stack, function_stack, program, will_be_nodes = self.add_node_random(arities_stack, function_stack, program, will_be_nodes, First, self.language.P, choice=choice)
            First = False

        return program

    
    def loss(self, constants, X, y):
        return ((self.evaluate(X, constants) - y)**2).mean()

    
    def optimize_constants(self, X, y):
        '''
        # This function optimizes the constants of the expression tree.
        '''
        if self.language.const_index not in self.traversal:
            return []
        
        x0 = np.abs(np.random.normal(0, 1, len(self.constants)))
        res = minimize(self.loss, x0, args=(X, y), method='BFGS', options={'disp': False})

        return res.x
