import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import LabelEncoder

def preprocessing():
    # Download the stopwords dataset
    nltk.download('stopwords')

    # Load the dataset into a dataframe
    df = pd.read_csv('IMDB_Dataset.csv')

    # Function for tokenization and removing punctuation and stopwords
    def tokenize_and_remove_punctuation(text):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove punctuation and stopwords
        tokens = [token.lower() for token in tokens if (token.isalpha() and token not in stopwords.words("english"))]
        
        return tokens

    # Apply preprocessing to each column
    for column in df.columns:
        df[column] = df[column].str.lower()  # Converting all to lowercase for consistency
        df[column] = df[column].str.replace('<.*?>', '', regex=True)  # Removing all HTML tags
        df[column] = df[column].apply(tokenize_and_remove_punctuation)

    # Convert tokenized words into strings
    df['processed_review'] = df['review'].apply(' '.join)

    # Ensure 'sentiment' column contains strings
    df['sentiment'] = df['sentiment'].apply(lambda x: x[0])  # Extracting the first (and only) element in the list

    return df
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df['processed_review'], df['sentiment'], test_size=0.2, random_state=42)

# # Create a Bag of Words (BoW) representation
# vectorizer = CountVectorizer()
# X_train_bow = vectorizer.fit_transform(X_train)
# X_test_bow = vectorizer.transform(X_test)

# # Convert sentiment labels to numerical values
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)

# # Train a Naive Bayes classifier
# classifier = MultinomialNB()
# classifier.fit(X_train_bow, y_train_encoded)

# # Convert sentiment labels in the test set to numerical values
# y_test_encoded = label_encoder.transform(y_test)

# # Make predictions on the test set
# predictions = classifier.predict(X_test_bow)


# # Evaluate the model
# accuracy = accuracy_score(y_test_encoded, predictions)
# print(f'Accuracy: {accuracy:.2f}')

# # Display additional metrics
# print(classification_report(y_test_encoded, predictions))
