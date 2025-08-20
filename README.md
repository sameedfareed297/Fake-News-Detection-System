# Fake-News-Detection-System
'''pandas as pd: Used for data manipulation and working with tabular data (DataFrame). Core to loading, analyzing, and cleaning data.

numpy as np: Numerical operations — often used for arrays and matrices.

re: Regular Expressions — used for text cleaning (removing unwanted characters).

matplotlib.pyplot as plt: Visualization library — used for plotting graphs.

seaborn as sns: Built on top of matplotlib — provides more attractive and easy-to-read plots.

-----------------------------

fillna(''): Replaces any NaN (missing) values with empty strings ''.
Why: Your model can't handle missing text — this ensures there are no blanks that might break the preprocessing pipeline.
-----------------------------

df['content'] = df['author'] + ' ' + df['title']
Combines the 'author' and 'title' columns into a new column called 'content'.
 Why: Merges relevant textual information for better training. The model learns patterns from combined features.
-----------------------------

X = df.drop(columns='label', axis=1)
y = df['label']
X: All input features (everything except the label).
y: Target variable — whether the news is fake (1) or real (0).
 Why: You separate inputs (X) and output (y) to train your ML model.

-----------------------------

Using axis=0 is suitable when:

- You want to combine DataFrames with the same column structure (i.e., same column names).
- You want to append rows from one DataFrame to another.

If you want to concatenate DataFrames along columns (horizontally), you would use axis=1.

------------------------------

data = data.sample(frac = 1)
Randomizes the order of the rows to avoid model bias.

------------------------------

data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)
Resets the row indices and removes the old index column.
When you concatenated the DataFrames using pd.concat(), the resulting DataFrame data_merge might have had duplicate index values. This is because each DataFrame being concatenated had its own index, and pd.concat() didn't automatically reset the index.

--------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
🔍 Converts text into numerical vectors based on word importance:

fit_transform: learns vocabulary from training data and transforms it.

transform: applies same transformation on test data.

---------------------------------

“NLP is all about enabling machines to understand human language. In our project, we used text preprocessing, vectorization (TF-IDF), and classification to teach the machine how to distinguish real and fake news — which is a classic NLP use case.” '''