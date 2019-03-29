# Train a modified skip-gram model to derive CWS-optimized word embeddings,
# and store cosine similarity scores for co-occurring word pairs.
# Try 'chmod a+x get_emb.sh' if permission denied.

if [ ! $# == 3 ]; then
	echo "Usage: $0 <name> <infile-dir> <window-size> ";
	echo "e.g.: $0 try data/try_full_base.txt 4"
	exit 1;
fi

name=$1
data=$2
window=$3

python3 get_w2v.py $name $data $window
