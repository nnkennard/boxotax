export flag_string=`cat configs/config0000.txt | awk 'NF' | \
	grep -v \# | sed 's/^/ --/' | tr '\n' ' '`
python train.py $flag_string
