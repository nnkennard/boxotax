# Make directory structure
mkdir ../data
cd ../data
mkdir -p small/fma2snomed small/snomed2nci small/fma2nci
mkdir mappings readmes large_originals

echo "Directory structure complete"

# Download data
wget https://www.cs.ox.ac.uk/isg/projects/SEALS/oaei/2018/LargeBio_dataset_oaei2018.zip
unzip LargeBio_dataset_oaei2018.zip

dsds

echo "Download complete"

# Rename files
mv oaei_FMA_small_overlapping_nci.owl small/fma2nci/fma.owl
mv oaei_NCI_small_overlapping_fma.owl small/fma2nci/nci.owl
mv oaei_SNOMED_small_overlapping_nci.owl small/snomed2nci/snomed.owl
mv oaei_NCI_small_overlapping_snomed.owl small/snomed2nci/nci.owl
mv oaei_FMA_small_overlapping_snomed.owl small/fma2snomed/fma.owl
mv oaei_SNOMED_small_overlapping_fma.owl small/fma2snomed/snomed.owl

# Move unused stuff away
mv ./*map* mappings
mv ./*.owl large_originals
mv ./*.txt readmes
mv ./*.rdf readmes

echo "Data moves complete"

cd ../scripts/

# Streamline data and make train test splits
python streamline_and_split.py ../data

 