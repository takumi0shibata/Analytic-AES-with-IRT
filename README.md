# Analytic Automated Essay Scoring Model incorporating Multidimensional IRT
This is the code of the paper _Analytic_ _Automated_ _Essay_ _Scoring_ _based_ _on_ _Deep_ _Neural_ _Networks_ _Integrating_ _Multidimensional_ _Item_ _Response_ _Theory_ published in COLIG2022.

## Requirements
- python = 3.8.10
- tensorflow = 2.9.2
- numpy = 1.21.6
- pandas = 1.3.5
- nltk = 3.7

## Training
If you want to use word series as an input, you should download `glove.6B.50d.txt` from https://nlp.stanford.edu/projects/glove/ and place in the `embeddings` folder.

The following code executes 5-fold cross validation.
```
bash CV_CTS.sh
bash CV_Proposed.sh
```

Note that the bash file implements each of the five different seeds for each of the word and pos inputs.

## Ackowledgement
Codebase from [cross-prompt-trait-scoring](https://github.com/robert1ridley/cross-prompt-trait-scoring)
