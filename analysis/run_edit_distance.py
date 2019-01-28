import subprocess
import sys

DATASET_PAIRS =["GO-HPO", "GO-MSH", "GO-OMIM", "HGNC-MSH", "HGNC-OMIM",
"HPO-MSH", "HPO-OMIM", "HPO-RXNORM", "MSH-OMIM", "MSH-RXNORM", "OMIM-RXNORM"]

GET_CIDS_STRING = ("cat $MAPPING_PATH/$PAIR.tsv | awk '{print $NF}' |"
    "tr ':' '\\t' | awk '{print $NF}' | sort | uniq > $OUTPUT_DIR/$PAIR.cids")
GREP_LINES_STRING = ("grep -F -f $OUTPUT_DIR/$PAIR.cids $MRCONSO > "
    "$OUTPUT_DIR/MRCONSO_$PAIR.RRF")
ED_DIST_STRING = ("python two_source_edit_distance.py $MAPPING_PATH/$PAIR.tsv "
    "$OUTPUT_DIR/MRCONSO_$PAIR.RRF > $OUTPUT_DIR/$PAIR.edd")

def get_command_strings(mapping_path, pair, output_dir, mrconso_path):
  return[cmd_str.replace("$MAPPING_PATH", mapping_path).replace("$MRCONSO",
    mrconso_path).replace("$OUTPUT_DIR", output_dir).replace("$PAIR", pair)
    for cmd_str in [
      GET_CIDS_STRING, GREP_LINES_STRING, ED_DIST_STRING]]

def main():
  mapping_path, output_dir, mrconso_path = sys.argv[1:]

  for pair in DATASET_PAIRS:
    print('echo "'+pair+'"')
    cmds = get_command_strings(mapping_path, pair, output_dir, mrconso_path)
    for cmd in cmds:
      print(cmd)

if __name__ == "__main__":
  main()
