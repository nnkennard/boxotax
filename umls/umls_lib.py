DB_FILE = "umls.db"

CONCEPT_FILE = "META/MRCONSO.RRF"
RELATION_FILE = "META/MRREL.RRF"
SEMTYPE_FILE = "META/MRSTY.RRF"

ALL_RELATION_TYPES = ["RO", "RB", "RN", "PAR", "CHD"]

ALL_SOURCES = ["AIR", "AOD", "AOT", "ATC" "CCS",
"CCS_10", "CSP", "CST", "CVX", "FMA", "GO", "HCPCS", "HGNC", "HL7V2.5", "HL7V3.0",
"HPO", "ICD10PCS", "ICD9CM", "ICPC", "LCH_NW", "LNC", "MEDLINEPLUS", "MSH",
"MTH", "MTHHH", "MTHICD9", "MTHMST", "MTHSPL", "MVX", "NCBI",  "NCI", "NDFRT",
"NDFRT_FDASPL", "NDFRT_FMTSME", "OMIM", "PDQ", "RAM", "RXNORM", "SOP", "SRC",
"TKMT", "USPMG", "UWDA", "VANDF"]

def read_file(filename):
  with open(filename, 'r') as f:
    for line in f:
      fields = line.strip().split('|')
      yield fields
