# Make directory structure
mkdir ../data
cd ../data
mkdir -p small/fma2snomed small/snomed2nci small/fma2nci
mkdir -p large/
mkdir mappings readmes

echo "Directory structure complete"

# Download data
wget https://www.cs.ox.ac.uk/isg/projects/SEALS/oaei/2018/LargeBio_dataset_oaei2018.zip
unzip LargeBio_dataset_oaei2018.zip

echo "Download complete"

# Rename files
mv oaei_FMA_small_overlapping_nci.owl small/fma2nci/fma.owl
mv oaei_NCI_small_overlapping_fma.owl small/fma2nci/nci.owl
mv oaei_SNOMED_small_overlapping_nci.owl small/snomed2nci/snomed.owl
mv oaei_NCI_small_overlapping_snomed.owl small/snomed2nci/nci.owl
mv oaei_FMA_small_overlapping_snomed.owl small/fma2snomed/fma.owl
mv oaei_SNOMED_small_overlapping_fma.owl small/fma2snomed/snomed.owl

mv oaei_FMA_whole_ontology.owl large/fma.owl
mv oaei_NCI_whole_ontology.owl large/nci.owl
mv oaei_SNOMED_extended_overlapping_fma_nci.owl large/snomed.owl

# Move unused stuff away
mv ./*map* mappings
mv ./*.txt readmes
mv ./*.rdf readmes

echo "Data moves complete"

