# hardware_classification

conda create -yn hardware_classification python=3.10
conda activate hardware_classification
pip install -r requirement.txt

This is just a simplified demo: the normal process should be like:

labelled raw data -> 
regex filter (or other information extraction) ->
bert tokenizer ->
bert encoder ->
simple classification model -> predicted class