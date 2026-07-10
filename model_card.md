# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This is a Random Forest classifier, built using scikit-learn's `RandomForestClassifier`. It was trained with a fixed random seed (`random_state=42`) so the results can be reproduced. I used the default settings for the model itself, without extra tuning. Before training, the categorical (text-based) features are converted into numbers using one-hot encoding, and the target label is converted into 0s and 1s using a label binarizer. The trained model, encoder, and label binarizer are all saved to disk as pickle files so they can be reused later without retraining.

## Intended Use
This model predicts whether someone's income is above or below $50,000 per year, based on information from the U.S. Census. It was built as a learning project for a Udacity course on deploying machine learning models with FastAPI. It's meant for practice and demonstration purposes only — it should not be used to make real decisions about people, such as hiring, lending, or insurance, for the reasons explained below.

## Training Data
The model was trained on the UCI Census Income dataset (sometimes called the "Adult" dataset). It has about 32,500 rows, each representing one person from the 1994 U.S. Census, along with whether they earned more or less than $50,000. The data includes both numbers (like age and hours worked per week) and categories (like education, marital status, occupation, race, and sex). Some entries in three columns — workclass, occupation, and native country — are missing and marked with a "?" instead of being left blank. Rather than removing or guessing these missing values, they were just treated as their own category. Before training, the data was split so that 80% was used for training and 20% was set aside for testing.

## Evaluation Data
The model was tested on that same 20% of data that was held back during training. This test data was processed the same way as the training data, using the same encoder, so everything lines up correctly.

## Metrics
To check how well the model works, I used three common metrics: precision, recall, and F1 score. Here's what the model scored on the test data:

- **Precision: 0.7419**
- **Recall: 0.6384**
- **F1: 0.6863**

In plain terms: when the model predicts someone earns over $50K, it's right about 74% of the time (precision). Out of everyone who actually earns over $50K, the model correctly identifies about 64% of them (recall).

I also checked the model's performance separately for each group within every category (like each education level, each race, etc.) — the full results are in `slice_output.txt`. The scores bounce around a lot for small groups (sometimes hitting a perfect 1.0 or a 0.0), simply because there aren't many examples to test on for those groups.

## Ethical Considerations
This data is from 1994, so it reflects the economy and society of that time — it shouldn't be assumed to still be accurate today. More importantly, the model doesn't perform equally well for everyone. For example, it's much worse at correctly identifying higher earners among people who are widowed (only catches about 16% of them) or who list "own child" as their relationship status (about 18%), compared to its overall rate of 64%. It also does poorly for people with a 7th-8th grade education, catching 0% of higher earners in that group. Since the model also uses race, sex, and country of origin directly as inputs, it risks repeating unfair patterns that existed in the original data. Because of all this, this model shouldn't be used to make real decisions about real people.

## Caveats and Recommendations
- This data is over 30 years old and doesn't reflect today's economy or population.
- Missing values (the "?" entries) were kept as their own category instead of being cleaned up or filled in — a different approach here might change the results.
- Be cautious about performance numbers for small groups (like certain countries with very few examples in the test set) — those numbers aren't very reliable.
- I didn't do any hyperparameter tuning — just used the model's default settings. Trying different settings, or a different type of model, could likely improve results.
- If this model were ever going to be used for something real, it would need to be retrained on more current, representative data first, and the fairness issues above would need to be addressed.