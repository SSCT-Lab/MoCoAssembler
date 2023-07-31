import random

from MoCo import MoCo
import mutate
from mutate import Mutator
from depart import Single_Model, Departed_Model
from assemble_complex import Assembler_Complex
import complex_models


class MoCoPT(MoCo):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.m = Mutator()
        self.n = 5
        self.model_name = model_name

    def depart(self):
        return complex_models.get_seed_model(model_name=self.model_name)

    def mutate(self):
        c = random.choice([0, 1])
        if c == 0:
            return self.mutate_on_parma
        else:
            return self.mutate_on_function

    def generate_model(self):
        a = Assembler_Complex(self.model_name)
        a.set_n(self.n)
        a.assemble_code_tree()

    def get_function(self, line: str):
        return mutate.get_function(line)

    def get_params(self, line: str):
        return mutate.get_params(line)

    def generate_param_line(self, line: str, params_dict: dict) -> str:
        return mutate.generate_line(line, params_dict)

    def random_param(self, data) -> str:
        return mutate.get_value(data)

    def mutate_on_param(self, line: str, func_file: str) -> str:
        return self.m.api_para_mutate(line)

    def mutate_on_function(self, line: str, func_file: str) -> str:
        return self.m.api_name_mutate(line)

    def mutate_on_module(self, function: str, Inception: dict) -> str:
        return '...'
