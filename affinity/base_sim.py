import nltk
import re
import sys

from nltk.corpus import wordnet
from nltk.corpus import stopwords, wordnet
from nltk.metrics import edit_distance
from nltk.stem.snowball import SnowballStemmer

CAMEL_CASE_MATCHER = re.compile("([A-Z]?[a-z]*)([A-Z][a-z]*)*")
ENGLISH_STOPWORDS = stopwords.words('english')
STEMMER = SnowballStemmer("english")

def normalize(input_string):
  # TODO : All caps
  if "ROOT" in input_string:
    return [input_string]
  if "_" in input_string:
    return input_string.split("_")
  else:
    match = re.match(CAMEL_CASE_MATCHER, input_string)
    if match is not None:
      return re.sub(
          '(?!^)([A-Z][a-z]+)', r' \1', input_string).split()
    else:
      return input_string.split()


def get_content(tokens):
  return [STEMMER.stem(token.lower())
      for token in tokens if token.lower() not in ENGLISH_STOPWORDS]


def common_word_score(s_tokens, t_tokens):
  denom = len(s_tokens) + len(t_tokens)
  common_words = set(s_tokens).intersection(set(t_tokens))
  return 2.0 * len(common_words) / denom

def base_sim(label_s, label_t):

  if label_s == label_t:
    return 1.0
  else:
    s_tokens, t_tokens = normalize(label_s), normalize(label_t)

  if s_tokens == t_tokens:
    return 1.0
  else:
    s_content_tokens, t_content_tokens = (get_content(s_tokens),
        get_content(t_tokens))

  if s_content_tokens == t_content_tokens:
    return 1.0
  else:
    return common_word_score(s_content_tokens, t_content_tokens)

def main():

  print(base_sim("my second questioning", "MyFirstQuestioning"))
  
if __name__ == "__main__":
  main()
