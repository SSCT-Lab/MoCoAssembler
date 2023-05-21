import spacy
from sentence_transformers import SentenceTransformer


class Similarity:
    nlp = spacy.load("en_core_web_sm")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def __init__(self): pass

    def levenshtein_distance(self, hypothesis: list, reference: list): pass

    def api_param_sim(self, param_json): pass

    def text_processing(self, sentence): pass

    def api_def_sim(self, def_json): pass

    def data_dumps(self): pass

    def sim_calcu(self, w_def: float, w_param: float): pass
