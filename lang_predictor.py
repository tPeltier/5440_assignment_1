from sys import exit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

print("preprocessing...")

eng_df = pd.read_csv("./english.txt", header=None, names=["word"])
eng_df["language"] = "english"
germ_df = pd.read_csv("./german.txt", header=None, names=["word"], encoding="latin-1")
germ_df["language"] = "german"
french_df = pd.read_csv("./french.txt", header=None, names=["word"], encoding="latin-1")
french_df["language"] = "french"
preprocessing_df = pd.concat([eng_df, germ_df, french_df]).reset_index(drop=True)

def prune_len(df, word_size):
    return df[df["word"].str.len() == word_size].reset_index(drop=True)

def add_ord_arr(df):
    df["ord"] = df["word"].apply(lambda word: [ord(char) for char in word])
    return df

def lang_to_label(df):
    langs = {lang: i for i, lang in enumerate(df["language"].unique())}
    df["label"] = df["language"].map(langs)
    return df

def str_to_ord_arr(word):
    return [ord(char) for char in word.lower()]

print(preprocessing_df.sample(10))

# FIX:ed? by use of str_to_ord_arr
# the txt files are lowercase but the test data is uppercase
# for now just upppering the input
# training_df["word"] = training_df["word"].str.upper()

preprocessing_df = prune_len(preprocessing_df, 5)
preprocessing_df = add_ord_arr(preprocessing_df)
preprocessing_df = lang_to_label(preprocessing_df)
print(preprocessing_df.sample(10))

testing_df = preprocessing_df.sample(frac=0.2)
print(testing_df.info())

training_df = preprocessing_df.drop(testing_df.index)
print(training_df.info())

training = np.array(training_df["ord"].tolist())
target = np.array(training_df["label"].tolist())
print(training)
print(target)

print("training...")

#NOTE: TUTORIAL PROVIDED CODE
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()
knn_model.fit(training, target)
svm_model.fit(training, target)
mlp_nn.fit(training, target)

print("predicting...")

# # WARN: DEMO/TEST PRINTS
# # Predicting “ANYONE”
# print(knn_model.predict([str_to_ord_arr("ANYONE")])) # Output: 0 (English)
# print(svm_model.predict([str_to_ord_arr("ANYONE")])) # Output: 0 (English)
# print(mlp_nn.predict([str_to_ord_arr("ANYONE")])) # Output: 0 (English)
#
# # Predicting “BÄRGET”
# print(knn_model.predict([str_to_ord_arr("BÄRGET")])) # Output: 1 (German)
# print(svm_model.predict([str_to_ord_arr("BÄRGET")])) # Output: 1 (German)
# print(mlp_nn.predict   ([str_to_ord_arr("BÄRGET")])) # Output: 1 (German)

# Note that predict() must also take a 2D array as our training data was a 2D array.

# TODO:
# - impl a func that gets a random word from a random lang and tests it
# - can test specific word or specific lang as well
# since i have all the data about the word, i can print the expected as well
# that is, print a nice log/test message
def test_n_words(testwords_df, n):
    test_words = testwords_df.sample(n)
    print(test_words)

test_n_words(testing_df, 5)

# TODO: GRAPH ACTUAL RESULTS

# #NOTE: TUTORIAL PROVIDED CODE
#
# # Label text for each graph
# labels = ("KNN", "SVM", "MLP")
#
# # Numbers that you want the bars to represent
# value = [81, 90, 71]
#
# # Title of the plot
# plt.title("Model Accuracy")
#
# # Label for the x values of the bar graph
# plt.xlabel("Accuracy")
#
# # Drawing the bar graph
# y_pos = np.arange(len(labels))
# plt.barh(y_pos, value, align="center", alpha=0.5)
# plt.yticks(y_pos, labels)
#
# # Display the graph
# plt.show()
