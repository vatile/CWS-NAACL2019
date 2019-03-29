# Create source file for the skip-gram model: word to index, and co-occurring word pairs
# Try 'chmod a+x get_src.sh' if permission denied

if [ ! $# == 3 ]; then
	echo "Usage: $0 <name> <data-dir> <window-size> ";
	echo "e.g.: $0 try data/try_full_base.txt 4"
	exit 1;
fi

name=$1
data=$2
window=$3

python3 preprocess.py $name $data $window
