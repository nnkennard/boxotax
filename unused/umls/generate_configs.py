
CHILDREN_STRING = "CHD|RN"

DISEASE_TYPE_PREFIX = ""
GENE_TYPE_PREFIX = ""

CONFIG_FILE_PATH = "/home/nnayak/boxotax/umls/configs.txt"

def config_number_format(config_number):
  return str(config_number).zfill(5)

def main():
  with open(CONFIG_FILE_PATH, 'r') as f:
    for line in f:
      number = line.split("\t")[0]
    next_number = int(number) + 1
  with open(CONFIG_FILE_PATH, 'a') as f:
    pass



if __name__=="__main__":
  main()
