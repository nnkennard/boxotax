# TODO: Add only probabilities from train set
# TODO: Use real similarity file
python 05a_add_inter_probabilities.py \
	../data/small/fma2nci/fma2nci.similarities \
	../data/small/fma2nci/fma2nci.vocab

cat ../data/small/fma2nci/fma.out.train.conditional \
	../data/small/fma2nci/nci.out.train.conditional \
	../data/small/fma2nci/fma2nci.conditional > \
	../data/small/fma2nci/fma2nci.train.conditional

cat ../data/small/fma2nci/fma.out.dev.conditional \
	../data/small/fma2nci/nci.out.dev.conditional >\
	../data/small/fma2nci/fma2nci.dev.conditional

