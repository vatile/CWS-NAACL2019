# CWS-NAACL2019
Code and data for the NAACL 2019 paper "Improving Cross-Domain Chinese Word Segmentation with Word Embeddings"

## Requirements: Python3.6, tf>=1.4.0, Keras==2.2.4

## Data Format
Please refer to the sample datafiles ('data/try_\*') for the required data format.
You need to first segment your raw training data with a baseline segmenter, which is not provided in this repository. However, baseline-segmented data used in the paper is provided under the 'data' path.

## Usage
You can simply run the whole segmentation model by run 'run.sh', after having prepared all required data in the proper format and modified parameters in 'run.sh'.

Or, you can run the model step-by-step using 'get_src.sh', 'get_emb.sh', and 'seg_file.sh'.

