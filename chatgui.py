import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

from keras.models import load_model
import json
import random
from tkinter import *

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

log = []

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                log.append(w)
                print(w)
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


# Creating GUI with tkinter

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#000000", font=("Verdana", 12))

        res = chatbot_response(msg)
        if res == "Thank you for your responses.":
            c = covid_prob()
            res += " There is a " + str(c) + " probability that you have COVID-19. Consult a health official and take an \nofficial test for a more reliable prediction. \n\nBot: How can I continue to assist you?"
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def covid_prob():
    #print(log)       #   0      1     2       3     4      5       6    7      8        9        10      11      12        13        14      15
    data = [0] * 16  # breath, fev, cough, throat, nose, asthma, lung, head, heart, diabetes, fatigue, travel, contact, gathering, public, masks = 0
    keywords = ["breathing", "problem", "fever", "dry", "cough", "sore", "throat", "nose", "asthma",
                "chronic", "lung", "headache", "heart", "diabetes", "diabetic", "fatigue", "travel", "travelled", "abroad",
                "contact", "unaware", "aware", "large", "gathering", "public", "exposed", "facemask", "mask"]
    match = [[0, 1], [0, 1], [1, 1], [2, 1], [2, 1], [3, 1], [3, 1], [4, 1], [5, 1], [6, 1], [6, 1], [7, 1], [8, 1], [9, 1], [9, 1], [10, 1], [11, 1], [11, 1], [11, 1], [12, 1], [12, -1], [12, -1], [13, 1], [13, 1], [14, 1], [14, 1], [15, 1], [15, 1]]
    for word in log:
        for i in range(len(keywords)):
            if keywords[i] == word:
                if match[i][1] == -1:  # if this is a word that cancels out the presence of another, set the spot in data to -1
                    data[match[i][0]] = -1
                if data[match[i][0]] != -1: # as long as this spot in data is not -1, then set it to 1
                    data[match[i][0]] = 1
    for i in range(len(data)): # go through an fix any that were turned to -1 (not possible to switch to 1) back to 0's so that the prediction works
        if data[i] == -1:
            data[i] = 0
    data = [data]

    print(data)
    model = pickle.load(open('COVID_model.sav', 'rb'))
    result = model.predict_proba(data)[0]
    result = round(result[1], 3)
    return result


base = Tk()
base.title("COVID Chatbot")
base.geometry("800x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="#add8e6", height="8", width="200", font="Arial",)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="20", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#3c9d9b',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="50", height="5", font="Arial")


# Place all components on the screen
scrollbar.place(x=780, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=770)
EntryBox.place(x=200, y=401, height=90, width=570)
SendButton.place(x=6, y=401, height=90)

ChatLog.config(state=NORMAL)
ChatLog.insert(END, "Bot: Hello, this program will assist in COVID prediction. \n\n")
ChatLog.insert(END, "Bot: Are you ready to begin? \n\n")
ChatLog.config(state=DISABLED)
ChatLog.yview(END)

'''data = [[1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]]
model = pickle.load(open('COVID_model.sav', 'rb'))
prob = model.predict_proba(data)[0]
print(prob[1])'''

base.mainloop()