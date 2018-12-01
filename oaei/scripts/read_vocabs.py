import sys

def main():
  pair_file = sys.argv[1]

  entities = set()

  with open(pair_file, 'r') as f:
    for line in f:
      _, _, hyper, hypo = line.strip().split()
      entities.update([hypo, hyper])

  with open(pair_file.replace(".out", ".vocab") , 'w') as f:
    for i, entity in enumerate(sorted(list(entities))):
      f.write("\t".join([entity, str(i)]) + "\n")

if __name__ == "__main__":
  main()
