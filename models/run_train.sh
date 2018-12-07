config_number=$1
config_file="configs/config"$config_number".txt"

export flag_string=`cat $config_file | awk 'NF' | \
	grep -v \# | sed 's/^/ --/' | tr '\n' ' '`
python train.py --config $config_number $flag_string
