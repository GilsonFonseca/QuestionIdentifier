
import os
import re
import nltk
import time
import pandas as pd
import files_ms_client
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


FILES_SERVER = os.environ.get("FILES_SERVER", "200.17.70.211:10162")

class QuestionsIdentifier:
    
    def __init__(self, specs):        
        self.specs = specs
        self.gb = GradientBoostingClassifier(n_estimators = 400, random_state=0)
        self.vectorizer = TfidfVectorizer(ngram_range=(1,3), 
                                    min_df=0.001, 
                                    max_df=0.7, 
                                    analyzer='word')
        self.question_pattern = ["do", "what", 'when', 'why', 'which', 'whom', 'where', 'whose', 'who', "is","would you", "how", "are","to know", "am i", 
                        "question", "tell", "can","answer", "ask"]
        self.question_types = ["whQuestion","ynQuestion"]

    def have_GPU(self):
        self.device_name = tf.test.gpu_device_name()
        if self.device_name == '/device:GPU:0':
            print('GPU encontrada em: {}'.format(self.device_name))
        else:
            print('GPU não encontrada')

    def configure_NLTK(self):
        nltk.download('punkt')
        nltk.download('nps_chat')
        self.posts = nltk.corpus.nps_chat.xml_posts()
        self.posts_text = [post.text for post in self.posts]

    def set_training(self):
        #divide train and test in 80 20
        train_text = self.posts_text[:int(len(self.posts_text)*0.8)]
        test_text = self.posts_text[int(len(self.posts_text)*0.2):]

        #Get TFIDF features
        self.X_train = self.vectorizer.fit_transform(train_text)
        self.X_test = self.vectorizer.transform(test_text)

        y = [post.get('class') for post in self.posts]

        self.y_train = y[:int(len(self.posts_text)*0.8)]
        self.y_test = y[int(len(self.posts_text)*0.2):]


    def execute_Gb(self):
        print("Training...")
        tmp = time.time()

        self.gb.fit(self.X_train, self.y_train)

        predictions_rf = self.gb.predict(self.X_test)

        # print(gb.feature_importances_)
        #Accuracy of 86%
        print(classification_report(self.y_test, predictions_rf))
        gpu_time = time.time() - tmp
        print("Training Time: %s seconds" % (str(gpu_time)))



    def is_predicted(self, question):
        question_type = self.gb.predict(self.vectorizer.transform([question]))
        return question_type in self.question_types

    def is_question(self, sentences):
        questionArray = []
        # delimiters = '[.?!]'
        # for sentence in sentences:
        #     question = sentence.lower().strip()
        #     phrase_arr = re.split(delimiters, question)
        for phrase in sentences:
            phrase = phrase.lower().strip()
            if phrase not in questionArray:
                if self.is_predicted(phrase):
                    questionArray.append(phrase)
                else:
                    if len(phrase.strip()):
                        first_word = nltk.word_tokenize(phrase)[0]
                        if phrase.endswith("?") or first_word in self.question_pattern:
                            questionArray.append(phrase)
        return questionArray


def main(msg):
    FILE = 'basebloom.xlsx'

    files_ms_client.download(msg["file"]["name"], FILE, url="http://" + FILES_SERVER)
    df = pd.read_excel(FILE)
    df.sample(frac=1)

    # Remove os dados N/A da base, assim como tabelas não usadas
    df = df.dropna()

    # Recebe df
    dataset = df.copy()
    sentences = dataset.Question.values

    QI = QuestionsIdentifier(msg)
    QI.have_GPU()
    QI.configure_NLTK()
    QI.set_training()
    QI.execute_Gb()
    questionArray = QI.is_question(sentences)

    for i, question in enumerate(questionArray):
        questionArray[i] = ' '.join((question).strip().split())

    qdataframe = pd.DataFrame({"questions":questionArray})
    qdataframe.to_excel("output.xlsx",
                    sheet_name='questions',
                    index=False)  
    
    msg["echo-file"] = files_ms_client.upload("output.xlsx", url="http://" + FILES_SERVER)