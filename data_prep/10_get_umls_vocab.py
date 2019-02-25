import box_lib
mport sys


def main():
  mrconso_file, source, output_dir = sys.argv[1:4]
  source_ab = box_lib.SAB_MAP[source]
  prefix = box_lib.PREFIX_MAP[source]
  vocab = {}
  with open(mrconso_file, 'r') as f:
    for line in f:
      fields = line.strip().split("|")
      (cui, term_status, string_type, sab, tty, code, label_str) = (
          fields[0], fields[2], fields[4], fields[11], fields[12], fields[13], fields[14])
      if not sab == source_ab:
        continue
      if source == "msh":
        if (term_status == "P" and string_type == "VCW" and tty in ["NM"]):
          if prefix + code not in vocab:
            vocab[prefix + code] = label_str
        if (term_status == "S" and string_type == "PF" and tty in ["MH", "NM",
        "ET", "TQ", "CE"]):
          if prefix + code not in vocab:
            vocab[prefix + code] = label_str
        if (term_status == "P" and string_type == "VO" and tty in ["MH", "NM"]):
          if prefix + code not in vocab:
            vocab[prefix + code] = label_str
        if (term_status == "P" and string_type == "VC" and tty in ["NM", "PM"]):
          if prefix + code not in vocab:
            vocab[prefix + code] = label_str
        if (term_status == "P" and string_type == "PF" and tty in ["NM", "PEP",
          "MH", "ET", "TQ", "PM"]):
          vocab[prefix + code] = label_str
        if (term_status == "P" and string_type == "VC" and tty in ["MH", "ET"]):
          vocab[prefix + code] = label_str
      elif source in ["go", "hpo"]:
        if term_status == 'S' and prefix + code not in vocab:
          if tty in ["ET", "OP"]:
            vocab[prefix + code] = label_str
        if tty in ["PT", "IS"]:
          vocab[prefix + code] = label_str

  output_file = output_dir + "/" + source + ".names"
  with open(output_file, 'w') as f:
    for k, v in vocab.items():
      f.write(k + "\t" + v + "\n")


if __name__ == "__main__":
  main()
