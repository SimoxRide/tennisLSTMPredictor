import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Lambda

def limit_output(x):
    # Limita il punteggio massimo per set a 7
    return tf.clip_by_value(x, 0, 7)


# Carica i dati (sostituisci con il tuo file CSV)
data = pd.read_csv("tennisdatasetprocessed.csv")

data["time"] = pd.to_datetime(data["time"])
data["time"] = data["time"].apply(lambda x: x.timestamp())
# Dividi i dati in set di addestramento e di valutazione
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Preprocessa i dati testuali (giocatori)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data["homeplayer"] + train_data["awayplayer"])
homeplayer_seq_train = tokenizer.texts_to_sequences(train_data["homeplayer"])
awayplayer_seq_train = tokenizer.texts_to_sequences(train_data["awayplayer"])
max_seq_length = max([len(seq) for seq in homeplayer_seq_train + awayplayer_seq_train])
homeplayer_seq_train = pad_sequences(homeplayer_seq_train, maxlen=max_seq_length, padding='post')
awayplayer_seq_train = pad_sequences(awayplayer_seq_train, maxlen=max_seq_length, padding='post')
homeplayer_seq_test = tokenizer.texts_to_sequences(test_data["homeplayer"])
awayplayer_seq_test = tokenizer.texts_to_sequences(test_data["awayplayer"])
homeplayer_seq_test = pad_sequences(homeplayer_seq_test, maxlen=max_seq_length, padding='post')
awayplayer_seq_test = pad_sequences(awayplayer_seq_test, maxlen=max_seq_length, padding='post')



# Preprocessa i dati numerici
num_features = ["time", "nset"]
train_numerical_data = train_data[num_features].values
test_numerical_data = test_data[num_features].values


# Prepara il target
train_target = train_data[["set1homescore", "set1awayscore", "set2homescore", "set2awayscore", "set3homescore", "set3awayscore", "set4homescore", "set4awayscore", "set5homescore", "set5awayscore"]].values
test_target = test_data[["set1homescore", "set1awayscore", "set2homescore", "set2awayscore", "set3homescore", "set3awayscore", "set4homescore", "set4awayscore", "set5homescore", "set5awayscore"]].values


# Definisci il modello
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 16


# Ramo testuale
homeplayer_input = Input(shape=(max_seq_length,), name="homeplayer_input")
awayplayer_input = Input(shape=(max_seq_length,), name="awayplayer_input")
homeplayer_emb = Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(homeplayer_input)
awayplayer_emb = Embedding(vocab_size, embedding_dim, input_length=max_seq_length)(awayplayer_input)
textual_data = Concatenate(axis=1)([homeplayer_emb, awayplayer_emb])
textual_lstm = LSTM(64)(textual_data)

# Ramo numerico
numerical_input = Input(shape=(len(num_features),), name="numerical_input")
numerical_dense = Dense(32, activation='relu')(numerical_input)

# Unisci i rami
#merged = Concatenate()([textual_lstm, numerical_dense])
#dense_1 = Dense(64, activation='relu')(merged)
#output = Dense(10, activation='linear')(dense_1)
merged = Concatenate()([textual_lstm, numerical_dense])
dense_1 = Dense(64, activation='relu')(merged)
output = Dense(10, activation='relu')(dense_1)
#limited_output = Lambda(limit_output)(output)
#outputs = [Lambda(limit_output)(output[:, i]) for i in range(10)]
#limited_output = tf.stack(outputs, axis=1)
limited_output = Lambda(limit_output)(output)



def format_prediction(prediction):
    formatted = []
    for i in range(0, len(prediction), 2):
        set_num = i // 2 + 1
        home_score = round(prediction[i])
        away_score = round(prediction[i + 1])
        formatted.append(f"Set {set_num} = {home_score} - {away_score}")

    return ", ".join(formatted)


model = Model(inputs=[homeplayer_input, awayplayer_input, numerical_input], outputs=limited_output)
#model = Model(inputs=[homeplayer_input, awayplayer_input, numerical_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mse','accuracy'])

# Allena il modello
model.fit([homeplayer_seq_train, awayplayer_seq_train, train_numerical_data], train_target, epochs=400, batch_size=32)

# Valuta il modello sui dati di valutazione
loss, mse,accuracy = model.evaluate([homeplayer_seq_test, awayplayer_seq_test, test_numerical_data], test_target)
accuracy=accuracy*100
print("Valutazione sul set di test - Loss: {:.2f}, MSE: {:.2f} , Accuracy: {:.2f}%".format(loss, mse,accuracy))

def get_user_input(tokenizer, max_seq_length):
    homeplayer = input("Inserisci il nome del giocatore di casa: ")
    awayplayer = input("Inserisci il nome del giocatore ospite: ")
    time_str = input("Inserisci la data e ora del match (YYYY-MM-DD HH:MM:SS): ")
    nset = int(input("Inserisci il numero di set: "))

    time_dt = pd.to_datetime(time_str)
    time_ts = time_dt.timestamp()

    homeplayer_seq = tokenizer.texts_to_sequences([homeplayer])
    awayplayer_seq = tokenizer.texts_to_sequences([awayplayer])
    homeplayer_seq = pad_sequences(homeplayer_seq, maxlen=max_seq_length, padding='post')
    awayplayer_seq = pad_sequences(awayplayer_seq, maxlen=max_seq_length, padding='post')

    numerical_data = np.array([[time_ts, nset]])

    return homeplayer_seq, awayplayer_seq, numerical_data

# Chiedi all'utente di inserire i dati
user_homeplayer_seq, user_awayplayer_seq, user_numerical_data = get_user_input(tokenizer, max_seq_length)

# Esegui la predizione sui dati forniti dall'utente
user_prediction = model.predict([user_homeplayer_seq, user_awayplayer_seq, user_numerical_data])
print(f"Predizione: {format_prediction(user_prediction[0])}")
model.save("Model.h5")
print(f"\n\nPrimi 10 risultati:")

#for i in range(0,10):
    #print(f"Predizione {i+1}: {format_prediction(user_prediction[i])}\n")




# Esegui la predizione sui dati di valutazione
#predictions = model.predict([homeplayer_seq_test, awayplayer_seq_test, test_numerical_data])
#print(f"Predizione: {format_prediction(predictions[0])}")


#for i, prediction in enumerate(predictions):
    #print(f"Predizione Set {i + 1}: {format_prediction(prediction)}")