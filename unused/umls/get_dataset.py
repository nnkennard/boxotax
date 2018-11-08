import sys
import umls_lib
import sqlite3

def main():
  conn = sqlite3.connect(umls_lib.DB_FILE)
  c = conn.cursor()

  config_file = sys.argv[1]
  output_file = config_file + ".out"

  with open(config_file, 'r') as f:
    lines = f.readlines()
    assert len(lines) == 1
    subgraph = umls_lib.read_subgraph_from_config_line(lines[0])
    pairs = subgraph.get_pairs(conn)
    with open(output_file, 'w') as out_file:
      out_file.write("\n".join(["\t".join(fields) for fields in pairs]))


if __name__=="__main__":
  main()
