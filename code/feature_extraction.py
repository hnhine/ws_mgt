import nltk
import textstat
import spacy
from nltk import word_tokenize, sent_tokenize
from lexical_diversity import lex_div as ld
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

def get_data(random_seed):
    """
    Load train, validation, and test datasets using the datasets library.
    """
    ds = load_dataset("Jinyan1/COLING_2025_MGT_en")
    train_df = ds['train'].train_test_split(shuffle = True, seed = random_seed, test_size='your size')['train']
    dev_df = ds['dev']    

    tmp = train_df.train_test_split(shuffle = True, seed = random_seed, test_size=0.1)
    train_df = tmp['train']
    val_df = tmp['test']
    
    return train_df, val_df, dev_df


# 1. Syntactic Complexity

def calculate_syntactic_complexity(text):
    doc = nlp(text)
    n_sentences = len(list(doc.sents))
    n_tokens = len([token for token in doc if not token.is_punct])
    n_noun_phrases = len(list(doc.noun_chunks))
    n_verbs = len([token for token in doc if token.pos_ == "VERB"])
    
    avg_sentence_length = n_tokens / n_sentences if n_sentences > 0 else 0
    avg_noun_phrases_per_sentence = n_noun_phrases / n_sentences if n_sentences > 0 else 0
    avg_verbs_per_sentence = n_verbs / n_sentences if n_sentences > 0 else 0

    syntactic_features = {
        "avg_sentence_length": avg_sentence_length,
        "avg_noun_phrases_per_sentence": avg_noun_phrases_per_sentence,
        "avg_verbs_per_sentence": avg_verbs_per_sentence,
    }
    return syntactic_features


# 2. Readability Metrics

def calculate_readability_metrics(text):
    readability_metrics = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "linsear_write_score": textstat.linsear_write_formula(text),
    }
    return readability_metrics

# 3. Lexical Diversity Metrics
print()
def compute_lexical_diversity(text):
    tokens = word_tokenize(text.lower())
    ttr = len(set(tokens)) / len(tokens)
    maas_ttr = ld.maas_ttr(tokens)
    hdd = ld.hdd(tokens)
    mltd = ld.mtld(tokens)
    return {
        'ttr': ttr,
        'maas_ttr': maas_ttr,
        'hdd': hdd,
        'mltd': mltd
    }

# 4. Text Statistics

def calculate_text_statistics(text):
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    
    difficult_words = [word for word in tokens if len(word) > 2 and textstat.syllable_count(word) > 2]
    num_difficult_words = len(difficult_words)
    unique_words = len(set(tokens))
    num_sentences = len(sentences)
    
    text_statistics = {
        'num_difficult_words': num_difficult_words,
        'unique_word_count': unique_words,
        'sentence_count': num_sentences,
    }
    return text_statistics

def extract_features(dataset):
    features = []
    for doc in tqdm(dataset['text'], desc="Extracting Features", unit="doc"):
        feature_vector = {}
        feature_vector.update(calculate_syntactic_complexity(doc))
        feature_vector.update(calculate_readability_metrics(doc))
        feature_vector.update(compute_lexical_diversity(doc))
        feature_vector.update(calculate_text_statistics(doc))
        features.append(feature_vector)
    return pd.DataFrame(features)



random_seed = 0
train_df, val_df, dev_df = get_data(random_seed)

train_features = extract_features(train_df)
val_features = extract_features(val_df)

train_features['label'] = train_df['label']
val_features['label'] = val_df['label']
train_features.to_json("train_features.jsonl", orient='records', lines=True)

val_features.to_json("val_features.jsonl", orient='records', lines=True)
# Train Gradient Boosting Classifier
X_train = train_features.drop(columns=['label'])
y_train = train_features['label']
X_val = val_features.drop(columns=['label'])
y_val = val_features['label']

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_seed)
clf.fit(X_train, y_train)

# Validate the model
y_val_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))

model_filename = "gradient_boosting_model.pkl"
joblib.dump(clf, model_filename)
print(f"Model saved to {model_filename}")

dev_features = extract_features(dev_df)
dev_features.to_json("dev_features.jsonl", orient='records', lines=True)
y_dev_pred = clf.predict(dev_features)

print("Development Accuracy:", accuracy_score(dev_df['label'], y_dev_pred))
print("Development Classification Report:\n", classification_report(dev_df['label'], y_dev_pred))



# Example output
# Syntactic Complexity: {'avg_sentence_length': 9.0, 'avg_noun_phrases_per_sentence': 2.0, 'avg_verbs_per_sentence': 1.0}
# Readability Metrics: {'flesch_reading_ease': 68.1, 'flesch_kincaid_grade': 8.0, ...}
# Lexical Diversity Metrics: {'ttr': 0.8, 'maas_ttr': 0.7, 'hdd': 0.72, 'mltd': 0.65}
# Text Statistics: {'num_difficult_words': 5, 'unique_word_count': 10, 'sentence_count': 1}

# Note: Make sure you have installed the required libraries using the following commands:
# !pip install spacy
# !pip install textstat
# !pip install lexical_diversity
# !pip install scikit-learn
# !python -m spacy download en_core_web_sm
