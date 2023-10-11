import re
import nltk
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('nps_chat')

question_pattern = ["do", "wh", "is","would you", "how", "are","to know", "am i", 
                   "question", "tell", "can","answer", "ask"]

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
  print('GPU encontrada em: {}'.format(device_name))
else:
  raise SyntaxError('GPU não encontrada')

posts = nltk.corpus.nps_chat.xml_posts()


posts_text = [post.text for post in posts]

#divide train and test in 80 20
train_text = posts_text[:int(len(posts_text)*0.8)]
test_text = posts_text[int(len(posts_text)*0.2):]

#Get TFIDF features
vectorizer = TfidfVectorizer(ngram_range=(1,3), 
                             min_df=0.001, 
                             max_df=0.7, 
                             analyzer='word')

X_train = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

y = [post.get('class') for post in posts]

y_train = y[:int(len(posts_text)*0.8)]
y_test = y[int(len(posts_text)*0.2):]

# Fitting Gradient Boosting classifier to the Training set
gb = GradientBoostingClassifier(n_estimators = 400, random_state=0)
#Can be improved with Cross Validation

gb.fit(X_train, y_train)

predictions_rf = gb.predict(X_test)

# print(gb.feature_importances_)
#Accuracy of 86% not bad
print(classification_report(y_test, predictions_rf))

question_types = ["whQuestion","ynQuestion"]

def is_predicted(question):
    question_type = gb.predict(vectorizer.transform([question]))
    return question_type in question_types

def is_question(question):
    is_ques = False
    questionArray = []
    question = question.lower().strip()
    sentence_arr = re.split('[.?!]', question)
    for sentence in sentence_arr:
        if sentence not in questionArray:
            if is_predicted(sentence):
                questionArray.append(sentence)

            if len(sentence.strip()):
                first_word = nltk.word_tokenize(sentence)[0]
                if sentence.endswith("?") or first_word in question_pattern:
                    questionArray.append(sentence)
    return questionArray

questionArray = is_question('OK, so we are going to start talking about this course content now. What is this course about? It is about algorithms-- introduction to algorithms. Who would guess?')

for i, question in enumerate(questionArray):
    questionArray[i] = ' '.join((question + "?").strip().split())
    
print(questionArray)