import tensorflow as tf

class StackedLSTM(tf.keras.Model):
    def __init__(self, input_dim, embed_dim, output_dim, rnn_units, dropout_size = 0.2):
        super(StackedLSTM,self).__init__()
        self.embed = tf.keras.layers.Embedding(input_dim, embed_dim)
        self.lstm1 = tf.keras.layers.LSTM(rnn_units,return_sequences=True)
        self.drop1 = tf.keras.layers.Dropout(dropout_size)
        self.lstm2 = tf.keras.layers.LSTM(int(rnn_units/2),return_sequences=True)
        self.drop2 = tf.keras.layers.Dropout(dropout_size)
        self.fc = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.embed(inputs)
        x = self.lstm1(x)
        x = self.drop1(x)
        x = self.lstm2(x)
        x = self.drop2(x)
        return self.fc(x)
    



