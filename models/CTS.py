import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from custom_layers.zeromasking import ZeroMaskedEntries
from custom_layers.attention import Attention


def build_CTS(vocab_size, maxnum, maxlen, readability_feature_count, linguistic_feature_count, configs, output_dim, embedding_weights=None):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    input_essay = layers.Input(shape=(maxnum*maxlen,), dtype='int32', name='input_essay')
    linguistic_input = layers.Input((linguistic_feature_count,), name='linguistic_input')
    readability_input = layers.Input((readability_feature_count,), name='readability_input')

    x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=maxnum*maxlen,
                            weights=embedding_weights, mask_zero=True)(input_essay)
    x = ZeroMaskedEntries()(x)
    x = layers.Dropout(dropout_prob)(x)
    x = layers.Reshape((maxnum, maxlen, embedding_dim))(x)
    x = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'))(x)
    x = layers.TimeDistributed(Attention())(x)

    x_list = [layers.LSTM(lstm_units, return_sequences=True)(x) for _ in range(output_dim)]
    x_list = [Attention()(x) for x in x_list]
    x_list = [layers.Concatenate()([rep, linguistic_input, readability_input])
                                 for rep in x_list]
    x_list = tf.concat([layers.Reshape((1, lstm_units + linguistic_feature_count + readability_feature_count))(rep)
                                 for rep in x_list], axis=-2)

    final_preds = []
    for index, _ in enumerate(range(output_dim)):
        mask = np.array([True for _ in range(output_dim)])
        mask[index] = False
        non_target_rep = tf.boolean_mask(x_list, mask, axis=-2)
        target_rep = x_list[:, index:index+1]
        att_attention = layers.Attention()([target_rep, non_target_rep])
        attention_concat = tf.concat([target_rep, att_attention], axis=-1)
        attention_concat = layers.Flatten()(attention_concat)
        final_pred = layers.Dense(units=1, activation='sigmoid')(attention_concat)
        final_preds.append(final_pred)

    y = layers.Concatenate()([pred for pred in final_preds])

    model = tf.keras.Model(inputs=[input_essay, linguistic_input, readability_input], outputs=y)

    model.summary()

    model.compile(loss='mse', optimizer='rmsprop')

    return model