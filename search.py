import json
import joblib
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def write_json(prediksi, judul, confidence):
    response_json = {
      'code': 200,
      'kategori': prediksi,
      'judul': judul,
      'confidence': confidence
    }
    with open("result.json", "w") as f:
        json.dump(response_json, f)

    return json.dumps(response_json)

def write_batch_json(prediksi, judul):
    response_json = {
      'code': 200,
      'batch_kategori': prediksi,
      'batch_judul': judul
    }
    with open("result.json", "w") as f:
        json.dump(response_json, f)

    return json.dumps(response_json)

def predict_and_search(headline_input):
    model = joblib.load("SAGA_LR_elasticnet_optimized.pkl") #load pipeline
    def text_cleaning(judul):
        judul = judul.lower()                                                # casefolding
        judul = re.sub('\(.*?\) | \[.*?\]', ' ', judul)                      # kata di dalam kurung
        judul = re.sub('[%s]' % re.escape(string.punctuation), ' ', judul)   # punctuation
        judul = re.sub('[‘’“”…]', ' ', judul)                                # tanda kutip

        return judul

    factory = StopWordRemoverFactory()
    new_stopwords = ['yg', 'dgn', 'dlm', 'nya'] #in case ada judul yang pakai ini
    stopwords = factory.get_stop_words() + new_stopwords
    stopwords = factory.create_stop_word_remover()

    def stopwords_removal(judul):
        judul = stopwords.remove(judul)

        return judul

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    title = headline_input #take data in index i
    title = text_cleaning(title) #clean+fold
    title = stopwords_removal(title) #stopword remove
    title = stemmer.stem(title) #stem
    title = [title] #wrap in numpy format
    prediction = model.predict(title) #pipeline run -> tokenize vectorize tfidf then classify
    prediction_proba = model.predict_proba(title)
    print('title is ', title)
    print('prediction is ', prediction)
    print('confidence level is ', prediction_proba)
    pResult = prediction[0] #result in [] so take data out
    proba = prediction_proba[0] #pop
    proba = sorted(proba, reverse=True) #sort
    response = write_json(pResult, title[0], proba[0])
    result = "finish!"
    return result, response

def batch_predict(headline_input):
    model = joblib.load("SAGA_LR_elasticnet_optimized.pkl") #load pipeline
    list_title = []
    list_category = []
    def text_cleaning(judul):
        judul = judul.lower()                                                # casefolding
        judul = re.sub('\(.*?\) | \[.*?\]', ' ', judul)                      # kata di dalam kurung
        judul = re.sub('[%s]' % re.escape(string.punctuation), ' ', judul)   # punctuation
        judul = re.sub('[‘’“”…]', ' ', judul)                                # tanda kutip

        return judul

    factory = StopWordRemoverFactory()
    new_stopwords = ['yg', 'dgn', 'dlm', 'nya'] #in case ada judul yang pakai ini
    stopwords = factory.get_stop_words() + new_stopwords
    stopwords = factory.create_stop_word_remover()

    def stopwords_removal(judul):
        judul = stopwords.remove(judul)
        return judul

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    for pairs in headline_input:
        title = pairs["message"] #take title from key-value
        title = text_cleaning(title) #clean+fold
        title = stopwords_removal(title) #stopword remove
        title = stemmer.stem(title) #stem
        title = [title] #wrap in numpy format
        prediction = model.predict(title) #pipeline run -> tokenize vectorize tfidf then classify
        pResult = prediction[0] #result in [] so take data out
        list_category.append({'kategori': pResult})
        list_title.append({'judul': title[0]})

    response = write_batch_json(list_category, list_title)
    result = "finish!"
    return result, response