# Python3.6, Keras==2.2.4, tf>=1.4.0 
# Try 'chmod a+x run.sh' if permission denied

name=try  # an arbitrary name of the source file folder
data=data/try_full_base.txt # txt file segmented by the baseline segmenter; test data should be included
testdata=data/try_test.txt # raw text of the test data
window=4 # window size for decoding
beam=10 # initial beam size for decoding
maxword=5 # max word length for decoding

# Create source file for the skip-gram model: word to index, and co-occurring word pairs
python3 get_w2v.py $name $data $window

# Train a modified skip-gram model to derive CWS-optimized word embeddings,
# and store cosine similarity scores for co-occurring word pairs.
python3 preprocess.py $name $data $window

# Decode test file and store the segmented text
python3 decode.py $name $testdata $window $beam $maxword
