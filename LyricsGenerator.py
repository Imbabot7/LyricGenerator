import numpy as np
import time
import tensorflow as tf
from LSTM import StackedLSTM

class LyricsGenerator:
    def __init__(self):
        with open('data/lyrics.txt') as f:
           self.corpus = ''.join(f.readlines())
        self.vocab = np.array(sorted(set(self.corpus)))
        self.seq_length = 250
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 10000
        self.embed = 256
        self.rnn_units = 1000
        self.learning_rate = 1e-3
        self.num_epochs = 300
        self.vectorize()
        self.model = self.build()

    def split_input_target(self,chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    
    def vectorize(self):
        self.char_repr = {c:i for i, c in enumerate(self.vocab)}
        self.text_vector = np.array([self.char_repr[c] for c in self.corpus])
        self.char_dataset = tf.data.Dataset.from_tensor_slices(self.text_vector)
        self.sequences = self.char_dataset.batch(self.seq_length+1, drop_remainder=True)
        AUTOTUNE = tf.data.AUTOTUNE
        self.dataset = self.sequences.map(self.split_input_target, num_parallel_calls=AUTOTUNE).cache()
        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    
    def build(self):
        return StackedLSTM(input_dim=len(self.vocab),embed_dim=self.embed,rnn_units=self.rnn_units,output_dim=len(self.vocab))

    def train(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        '''
        self.model.compile(optimizer=optimizer,loss=loss_fn)
        checkpoint_filepath = 'checkpoints/checkpoint_model4' 
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='loss',
            mode = 'min',
            verbose=1,
            save_weights_only=True,
            save_best_only=True)
        self.history = self.model.fit(self.dataset,epochs=self.num_epochs,callbacks=[model_checkpoint_callback])
        '''
        @tf.function
        def train_on_batch(X,y):
            with tf.GradientTape() as tape:
                y_pred = self.model(X, training=True)
                loss = loss_fn(y, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
            return loss
        
        best_loss=10000
        for epoch in range(self.num_epochs):
            print(f"\nStart of Training Epoch {epoch+1}")
            start = time.time()
            losses = []
            for batch_idx, (x_batch, y_batch) in enumerate(self.dataset):
                losses.append(train_on_batch(x_batch,y_batch))
            end = time.time()
            print(f"\nTime taken for current epoch is {end-start} seconds")
            avg_loss = np.mean(np.array(losses))
            print(f"\nLoss: {avg_loss}\n" )
            if avg_loss < best_loss:
                self.model.save_weights('checkpoints/checkpoint1')
                best_loss = avg_loss
        
    def generate(self,prompt='hi',length=100,temperature=1):
        generated_text = ''
        prompt_vector = [self.char_repr[c] for c in prompt]
        prompt_tensor = tf.expand_dims(prompt_vector,axis=0)
        for _ in range(length):
            prediction = self.model(prompt_tensor)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction/temperature
            predicted = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()
            generated_text += self.vocab[predicted]
            prompt_tensor = np.append(prompt_tensor.numpy(),np.asarray(predicted))
            prompt_tensor = prompt_tensor[-self.seq_length:]
            prompt_tensor = tf.expand_dims(prompt_tensor, 0)
        
        punct = list(range(7)) + [19]
        while predicted not in punct:
            prediction = self.model(prompt_tensor)
            prediction = tf.squeeze(prediction,0)
            prediction = prediction/temperature
            predicted = tf.random.categorical(prediction, num_samples=1)[-1,0].numpy()
            prompt_tensor = tf.expand_dims([predicted], 0)
            generated_text += self.vocab[predicted]
        
        return prompt + generated_text

    
    def load_weights(self,path):
        self.model.load_weights(path)
    
    def load_model(self,path):
        self.model = tf.keras.models.load_model(path)
    
g = LyricsGenerator()
#g.train()