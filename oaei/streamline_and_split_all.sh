 python streamline_data.py ../oaei2018_data/small/fma2nci/fma.owl fma
 python streamline_data.py ../oaei2018_data/small/fma2nci/nci.owl nci
 python streamline_data.py ../oaei2018_data/small/fma2snomed/fma.owl fma
 python streamline_data.py ../oaei2018_data/small/fma2snomed/snomed.owl snomed
 python streamline_data.py ../oaei2018_data/small/snomed2nci/snomed.owl snomed
 python streamline_data.py ../oaei2018_data/small/snomed2nci/nci.owl nci

 python split_test_train.py ../oaei2018_data/small/fma2nci/fma.owl.pairwise
 python split_test_train.py ../oaei2018_data/small/fma2nci/nci.owl.pairwise
 python split_test_train.py ../oaei2018_data/small/fma2snomed/fma.owl.pairwise
 python split_test_train.py ../oaei2018_data/small/fma2snomed/snomed.owl.pairwise
 python split_test_train.py ../oaei2018_data/small/snomed2nci/snomed.owl.pairwise
 python split_test_train.py ../oaei2018_data/small/snomed2nci/nci.owl.pairwise

 
