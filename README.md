# sentimentNaiveBayes

This is an implementation of the naive bayes algorithm for simple sentiment classification. 

The dataset used is at: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz. 

The initial vocabulary of positive and negative words used is at: http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar

Three versions are experimented with.

1. Simple Naive Bayes- nb.py (Accuracies: train-94.75%, test-81.5%)
2. Naive Bayes with negation handling- nb-only_negation_handling.py (Accuracies: train-94.8125%, test-81.5% )
3. Naive Bayes with bigrams- nb-only_negation_handling.py (Accuracies: train-94.9375%, test-83.25%)
