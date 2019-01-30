import sys

def main():
  mrconso_file, source = sys.argv[1:3]
  vocab = {}
  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (cui, term_status, string_type, sab, tty, code, label_str) = (
          fields[0], fields[2], fields[4], fields[11], fields[12], fields[13], fields[14])
      if not sab == source:
        continue
      if source == "MSH":
        if (term_status == "P" and string_type == "PF" and tty=="NM"):
          print("\t".join([code, label_str, cui]))
      elif source in ["GO", "HPO"]:
        if tty =="PT":
          print("\t".join([code, label_str, cui]))


if __name__ == "__main__":
  main()
