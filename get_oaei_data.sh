mkdir oaei2018_data
cd oaei2018_data
wget https://www.cs.ox.ac.uk/isg/projects/SEALS/oaei/2018/LargeBio_dataset_oaei2018.zip
unzip LargeBio_dataset_oaei2018.zip
mkdir -p small/fma2snomed
mkdir -p small/snomed2nci
mkdir -p small/fma2nci
mv oaei_FMA_small_overlapping_nci.owl small/fma2nci/fma.owl
mv oaei_NCI_small_overlapping_fma.owl small/fma2nci/nci.owl
mv oaei_SNOMED_small_overlapping_nci.owl small/snomed2nci/snomed.owl
mv oaei_NCI_small_overlapping_snomed.owl small/snomed2nci/nci.owl
mv oaei_FMA_small_overlapping_snomed.owl small/fma2snomed/fma.owl
mv oaei_SNOMED_small_overlapping_fma.owl small/fma2snomed/snomed.owl
mkdir mappings
mv ./*map* mappings
mkdir large_originals
mv ./*.owl large_originals
mkdir readmes
mv ./*.txt readmes
mv ./*.rdf readmes
