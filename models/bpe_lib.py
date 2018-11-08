import re

class BytePairEncoder(object):
  def __init__(self, word_freq_map):
    self.word_freq_map = word_freq_map
    self.pairs = self.get_stats()
 

  def get_stats(self):
    pairs = collections.defaultdict(int)
    for word, freq in self.word_freq_map.iteritems():
      symbols = word.split()
      for symbol_pair in zip(symbols[:-1], symbols[1:]):
        pairs[symbol_pair] += freq
    return pairs

  def merge_vocab(self, pair):
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, freq in self.word_freq_map.iteritems():
      new_word = p.sub(''.join(pair), word)
      new_vocab[new_word] = freq
   self.word_freq_map = new_vocab

  def merge_top(self):
    best = max(

