from pathlib import Path
from nltk import word_tokenize as nltk_word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import pickle


# Load abbreviation list for sentence tokenization (same as LeNER-Br)
_current_dir = Path(__file__).parent
punkt_param = PunktParameters()
with open(_current_dir / "abbrev_list.pkl", "rb") as fp:
    abbrev_list = pickle.load(fp)
punkt_param.abbrev_types = set(abbrev_list)
_sentence_tokenizer = PunktSentenceTokenizer(punkt_param)


def get_sentence_tokenizer():
    return _sentence_tokenizer


def word_tokenize(text, language='portuguese'):
    return nltk_word_tokenize(text, language=language)
