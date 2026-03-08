import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

print("importing...")

eng_url = "https://raw.githubusercontent.com/tPeltier/5440_assignment_1/refs/heads/main/english.txt"
eng_df = pd.read_csv(eng_url, header=None, names=["word"])
eng_df["language"] = "english"

germ_url = "https://raw.githubusercontent.com/tPeltier/5440_assignment_1/refs/heads/main/german.txt"
germ_df = pd.read_csv(germ_url, header=None, names=["word"], encoding="latin-1")
germ_df["language"] = "german"

french_url = "https://raw.githubusercontent.com/tPeltier/5440_assignment_1/refs/heads/main/french.txt"
french_df = pd.read_csv(french_url, header=None, names=["word"], encoding="latin-1")
french_df["language"] = "french"
preprocessing_df = pd.concat([eng_df, germ_df, french_df]).reset_index(drop=True)

print("importing complete")

print("preprocessing...")

def prune_len(df, word_size):
    return df[df["word"].str.len() == word_size].reset_index(drop=True)

def add_ord_arr(df):
    df["ord"] = df["word"].apply(lambda word: [ord(char) for char in word])
    return df

def lang_to_label(df):
    # WARN: ADJUST DICT ACCORDING TO LANGS
    langs = {"english": 0, "german": 1, "french": 2}
    df["label"] = df["language"].map(langs)
    return df

preprocessing_df = prune_len(preprocessing_df, 5)
preprocessing_df = add_ord_arr(preprocessing_df)
preprocessing_df = lang_to_label(preprocessing_df)

print(preprocessing_df.sample(10))

# most of the 5 letter words are english, so we force an even % split with groupby ( english still dominates but less so )
testing_df = preprocessing_df.groupby("language").sample(frac=0.2, random_state=31) # gives seeded, reproducable randomness

# will ignore the fact that english dominates the 5 letter word category
#testing_df = preprocessing_df.sample(frac=0.2, random_state=31) # gives seeded, reproducable randomness

#testing_df = preprocessing_df.sample(frac=0.2) # uncomment for true randomness

training_df = preprocessing_df.drop(testing_df.index)

training = np.array(training_df["ord"].tolist())
target = np.array(training_df["label"].tolist())

print(preprocessing_df["language"].value_counts())

print("preprocessing complete")

print("training...")

#NOTE: TUTORIAL PROVIDED CODE
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()
knn_model.fit(training, target)
svm_model.fit(training, target)
mlp_nn.fit(training, target)

print("training complete")

def test_n_words(testwords_df, n):
    test_words = testwords_df.sample(n, random_state=31) # gives seeded, reproducable randomness
    #test_words = testwords_df.sample(n) # uncomment for true randomness
    correct_preds = {"knn": 0, "svm": 0, "mlp": 0}

    for _, word in test_words.iterrows():
        # Note that predict() must also take a 2D array as our training data was a 2D array.
        knn_pred = knn_model.predict([word["ord"]])
        svm_pred = svm_model.predict([word["ord"]])
        mlp_pred = mlp_nn.predict([word["ord"]])

        correct_preds["knn"] += int(knn_pred[0] == word["label"])
        correct_preds["svm"] += int(svm_pred[0] == word["label"])
        correct_preds["mlp"] += int(mlp_pred[0] == word["label"])

        print(f"predicting for word \"{word['word']}\" -> expected language = {word['language']}:{word['label']}")
        print(f"  knn prediction: {knn_pred}")
        print(f"  svm prediction: {svm_pred}")
        print(f"  mlp prediction: {mlp_pred}")

    return correct_preds

print("predicting...")
number_of_words_to_test = 100
correct_preds = test_n_words(testing_df, number_of_words_to_test)
print("predicting complete")

def print_accuracies(correct_preds, n):
    accuracies = {k: (v / n) * 100 for k, v in correct_preds.items()}
    print(f"Accuracy over {n} words:")
    print(f"  knn: {accuracies['knn']:.1f}%")
    print(f"  svm: {accuracies['svm']:.1f}%")
    print(f"  mlp: {accuracies['mlp']:.1f}%")

    return list(accuracies.values())

models_accuracies = print_accuracies(correct_preds, number_of_words_to_test)

plt.title("Model Accuracy")
plt.xlabel("Accuracy")
labels = ("KNN", "SVM", "MLP") # ylabels

y_pos = np.arange(len(labels))
plt.barh(y_pos, models_accuracies, align="center", alpha=0.5)
plt.yticks(y_pos, labels)
plt.show()
