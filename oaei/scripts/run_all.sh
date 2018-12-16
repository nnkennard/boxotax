#sh 00_get_oaei_data.sh
python 01_streamline_and_split.py ../data/
bash 02_generate_vocabs.sh
python 03_assign_probabilities.py ../data/
python 04_calculate_alignments.py ../data/
