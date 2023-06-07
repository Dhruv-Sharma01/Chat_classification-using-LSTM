
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import io
import requests

# Step 1: Prepare and load the labeled dataset
url = 'https://raw.githubusercontent.com/Dhruv-Sharma01/Chat_classification-using-LSTM/main/summer.csv'
read_data = requests.get(url).content

data = pd.read_csv(io.StringIO(read_data.decode('utf-8')))  
sentences = data['sentence']
labels = data['label']

# Step 2: Tokenize and pad the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Step 4: Convert labels to categorical format
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 5: Build the LSTM model
vocab_size = len(word_index) + 1
embedding_dim = 100
max_length = padded_sequences.shape[1]

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(LSTM(128))
model.add(Dense(4, activation='softmax'))  # Updated for the "Admin" category

learning_rate = 0.0009  # Decreased learning rate
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Step 6: Train the LSTM model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Rest of the code remains the same...


# Step 7: Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Set Accuracy: {accuracy*100:.2f}%")

# Step 8: Load and process the ne
u=input()
new_sentences = [u]
while u!='exit':
   u=input()
   new_sentences.append(u)

 # Example admin message added
new_sequences = tokenizer.texts_to_sequences(new_sentences)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_length)

predictions = model.predict(new_padded_sequences)
question_words = ['what', 'why', 'who', 'when', 'how','Kya','Kaun','Kab','Kyon','Kitna','Kitni','describe','explain','which']
verbs = ['can', 'could', 'will', 'would', 'shall', 'should', 'is','are', 'did', 'do']

for i in range(len(new_sentences)):
    sentence = new_sentences[i]
    prediction = predictions[i]

    # Check if the sentence contains a question mark
    if '?' in sentence:
        predictions[i] = 0
    elif '?' not in sentence:
        predictions[i][0] =   (predictions[i][0])*0.05
    if 'kyunki' in sentence.lower() or 'because' in sentence.lower():
       predictions[i][2]*=2

    # Check if the sentence starts with a question word
    for word in question_words:
        if sentence.lower().startswith(word):
            predictions[i] = 0
            break

    # Check if the first word of the sentence is a verb
    if sentence.split()[0].lower() in verbs:
        predictions[i] = 0

    # Check if the sentence contains admin keywords
    admin_keywords = ['admin', 'alert', 'announcement','reminder']  # Add more admin keywords as needed
    r=sentence.lower()
    r=r.split()
    print(r)
    for j in r:
        if j in admin_keywords:
            predictions[i]=3
            # print(True,i,predictions)

label_mapping = {0: "Question", 1: "General", 2: "Answer", 3: "Admin"}
# print(predictions)
for sentence, prediction in zip(new_sentences, predictions):
    if max(prediction)!=3:
     predicted_label = label_mapping[prediction.argmax()]
    else:
     predicted_label = label_mapping[3]
    # print([prediction.argmax()])
    print(f"Sentence: {sentence}")
    print(f"Predicted Label: {predicted_label}")
    print()
