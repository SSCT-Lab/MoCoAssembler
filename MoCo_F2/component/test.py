import copy

from _database import database
from Mutate import get_mutator
from Concrete import concrete
from ResultAnalyse import analyser


mutator = get_mutator("torch")
leNet = database.get_seed("LeNet")
googleNet = database.get_seed("googlenet")
concrete.set_model_name("testGoogleNet")
beiyong = copy.deepcopy(googleNet["inception"])
for i in range(100):
    mutate_result, _ = mutator.mutate({"inception": beiyong})
    new_child_model_name = list(mutate_result.keys())[0]
    if i > 2:
        googleNet.pop(list(googleNet.keys())[-1])
    googleNet["googlenet"][-5]["layer"] = new_child_model_name
    googleNet[list(mutate_result.keys())[0]] = mutate_result[list(mutate_result.keys())[0]]
    result = concrete.perform(googleNet, 0, i)
    result[0]["mutate type"] = _
    analyse_result = analyser.analyse_result(result)
    print(analyse_result, result[0])
