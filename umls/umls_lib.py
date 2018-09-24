DB_FILE = "umls.db"

CONCEPT_FILE = "META/MRCONSO.RRF"
RELATION_FILE = "META/MRREL.RRF"
SEMTYPE_FILE = "META/MRSTY.RRF"

all_sources = ["MSH", "RXNORM", "MTH", "MONDO"]

def read_file(filename):
  with open(filename, 'r') as f:
    for line in f:
      fields = line.strip().split('|')
      yield fields
