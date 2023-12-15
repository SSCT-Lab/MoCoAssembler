from MoCo import MoCo
import complex_models
import assemble_complex_v2
import mutate_v2


class MoCoJT(MoCo):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.model = self.depart()
        self.model_name = model_name
        self.N = 5
        self.mutator = mutate_v2.Mutator()

    def depart(self):
        return complex_models.get_seed_model(self.model_name)

    def mutate(self):
        return str(self) + "\nmutate? dont use this, use other mutate functions instead."

    def generate_model(self):
        a = assemble_complex_v2.Assembler_Complex(self.model_name)
        a.set_n(self.N)
        a.assemble_code_tree()
        return

    def get_function(self, line: str):
        return mutate_v2.get_function(line)

    def get_params(self, line: str):
        return mutate_v2.get_params(line)

    def generate_param_line(self, line: str, params_dict: dict) -> str:
        return mutate_v2.generate_line(self.get_function(line), params_dict)

    def random_param(self, data) -> str:
        return "?"

    def mutate_on_param(self, line: str, func_file: str) -> str:
        return self.mutator.api_para_mutate(line)

    def mutate_on_function(self, line: str, func_file: str) -> str:
        return self.mutator.api_name_mutate(line)

    def mutate_on_module(self, function: str, Inception: dict) -> str:
        return "?"


if __name__ == '__main__':
    s = MoCoJT('lenet')
    s.generate_model()
