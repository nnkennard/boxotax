CONCEPT_FILE = "MRCONSO.RRF"
RELATION_FILE = "MRREL.RRF"
SEMTYPE_FILE = "MRSTY.RRF"

def read_file(filename):
  with open(filename, 'r') as f:
    for line in f:
      fields = line.strip().split('|')
      yield fields
