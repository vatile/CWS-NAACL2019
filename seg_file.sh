# Decode test file
# Try 'chmod a+x seg_file.sh' if permission denied

if [ ! $# == 5 ]; then
	echo "Usage: $0 <name> <testdata-dir> <window-size> <beam-size> <max-word-size>";
	echo "e.g.: $0 try data/try_test.txt 4 10 5"
	exit 1;
fi

name=$1
testdata=$2
window=$3
beam=$4
maxword=$5

python3 decode.py $name $testdata $window $beam $maxword
