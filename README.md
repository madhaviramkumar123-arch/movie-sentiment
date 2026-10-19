# Movie Review Sentiment Classifier

## Project Overview
A simple machine learning model that classifies movie reviews as positive or
negative. Review text is converted to numbers with TF-IDF, then two models
are compared: a Decision Tree and K-Nearest Neighbors (KNN).

## Dataset
Movie Review Data (Pang & Lee, Cornell), 2,000 labeled reviews
(1,000 positive, 1,000 negative):
https://www.cs.cornell.edu/people/pabo/movie-review-data/

Loaded through NLTK's `movie_reviews` corpus. No manual download is needed.

## Setup Instructions
1. Clone this repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install pandas scikit-learn matplotlib nltk jupyter`
3. Open the notebook: `jupyter notebook sentiment_classifier.ipynb`
4. Run all cells (the dataset downloads automatically via NLTK).

The notebook can also be run in Google Colab, where the libraries come
preinstalled.

## Results
| Model | Accuracy | Negative recall | Positive recall |
|-------|----------|-----------------|-----------------|
| Decision Tree (max_depth=20) | 67% | 0.61 | 0.73 |
| KNN (k=5, cosine distance) | 66% | 0.61 | 0.71 |
| Baseline (random guess) | 50% | - | - |

## Conclusion
Both the models Decision Tree and KNN beat the 50% random guess baseline but not really accurate on their own either. The lower values are mostly because the two models are weak when it comes to text. Additionally, TD-IDF counts words but neglects word order, thus 'not bad' would be marked as bad (negative sentiment) and 'not good' as 'good'. The average review is also rather long, with winded explanations on cast, plot devices and director. The buzzwords might be detected in parts of the review that aren't an opinion on the movie itself. Further, 1600 reviews is too little for language. More data helps texts models a lot especially to deal with sarcasm and hyperbole.
Decision Tree has an accuracy of 67.25% and KNN has an accuracy if 66%. Making Decision Tree slightly better. They still misread about a third of the reviews. Both models over-predicted 'positive' with negative reviews harder to identify (recall of only 0.61)
I have learnt that TD-IDF converts text into numbers, also how to read a confusion matrix and recall. I've also learnt that these models counting words struggle with negation such as 'not bad'. My next step to improvise would be to try models such as Logistic Regression and Naive Bayes which work better with text.
