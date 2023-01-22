import custom_layers.attention
import custom_layers.zeromasking
import custom_layers.mgpcm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers, regularizers, models

def build_ProposedModel(X_inputs, configs, max_sentnum, max_sentlen, num_item, num_category, vocab_size, latent_dim, inf_mask, beta_mask, weights=None, word=True):
    embedding_dim = configs.EMBEDDING_DIM
    dropout_prob = configs.DROPOUT
    cnn_filters = configs.CNN_FILTERS
    cnn_kernel_size = configs.CNN_KERNEL_SIZE
    lstm_units = configs.LSTM_UNITS

    input_essay = layers.Input(shape=(max_sentnum*max_sentlen,), dtype='int32')
    input_linguistic_features = layers.Input(shape=(X_inputs[1].shape[1],), dtype='float32')
    input_readability_features = layers.Input(shape=(X_inputs[2].shape[1],), dtype='float32')

    if word:
        x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=max_sentnum*max_sentlen,
                            weights=weights, mask_zero=True, trainable=True)(input_essay)
    else:
        x = layers.Embedding(output_dim=embedding_dim, input_dim=vocab_size, input_length=max_sentnum*max_sentlen,
                             weights=None, mask_zero=True, trainable=True)(input_essay)
    x = custom_layers.zeromasking.ZeroMaskedEntries()(x)
    x = layers.Dropout(dropout_prob)(x)
    x = layers.Reshape((max_sentnum, max_sentlen, embedding_dim))(x)
    x = layers.TimeDistributed(layers.Conv1D(cnn_filters, cnn_kernel_size, padding='valid'))(x)
    x = layers.TimeDistributed(custom_layers.attention.Attention())(x)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)
    x = custom_layers.attention.Attention()(x)
    x = layers.Concatenate(axis=-1)([x, input_linguistic_features, input_readability_features])

    theta = layers.Dense(latent_dim, name='theta',
                         activity_regularizer=regularizers.l2(0.01))(x)

    mgpcm = custom_layers.mgpcm.MGPCM(num_item=num_item, latent_dim=latent_dim, num_category=num_category,
                                      inf_mask=inf_mask, beta_mask=beta_mask, name='mgpcm_layer')

    output = mgpcm(theta)

    model = models.Model([input_essay,
                          input_linguistic_features,
                          input_readability_features],
                          output)
    model.summary()
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy')

    return model