export LC_ALL="C"
cat ../data/small/fma2nci/fma.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/fma2nci/fma.out.vocab

cat ../data/small/fma2nci/nci.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/fma2nci/nci.out.vocab

cat ../data/small/snomed2nci/nci.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/snomed2nci/nci.out.vocab

cat ../data/small/snomed2nci/snomed.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/snomed2nci/snomed.out.vocab

cat ../data/small/fma2snomed/snomed.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/fma2snomed/snomed.out.vocab

cat ../data/small/fma2snomed/fma.out.train |\
       	awk '{print $3"\n"$4}' | sort | uniq >\
	../data/small/fma2snomed/fma.out.vocab

