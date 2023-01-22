################
# import module
################
import nltk
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import random
import argparse
from tensorflow.keras.utils import to_categorical
from configs.configs import Configs
from utils.read_data import read_word_vocab, read_essays_words_cv, read_pos_vocab, read_essays_pos_cv
from utils.general_utils import get_scaled_down_scores, pad_hierarchical_text_sequences, load_word_embedding_dict, build_embedd_table, \
                                get_overall_score_range, get_analytic_score_range, get_min_max_scores, get_min_max_score_vector, get_attribute_mask_vector
from models.Proposed_2way import build_Proposed_2way
from models.Proposed_2way_noatt import build_Proposed_2way_noatt
from utils.evaluator import evaluator

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def main():
    parser = argparse.ArgumentParser(description='Proposed_upgrade_model')
    parser.add_argument('--prompt_id', type=int, default=1, help='prompt id of train/test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--word_input', type=bool, default=True, help='word input or pos tag input')
    parser.add_argument('--prediction_option', type=str, default='ex', help='set prediction option: ex or argmax')
    parser.add_argument('--latent_dim', type=int, default=1, help='set latent dim')
    parser.add_argument('--with_attn', type=bool, default=False, help='use Attention?')

    args = parser.parse_args()
    id = args.prompt_id
    seed = args.seed
    word = args.word_input
    pred_opt = args.prediction_option
    dim = args.latent_dim
    with_attn = args.with_attn
    ###################
    # Set Parameters
    ###################
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    configs = Configs()

    data_path = configs.DATA_PATH1
    features_path = configs.FEATURES_PATH
    readability_path = configs.READABILITY_PATH
    vocab_size = configs.VOCAB_SIZE
    embedding_path = configs.EMBEDDING_PATH
    Fold = configs.FOLD
    EPOCHS = configs.EPOCHS
    BATCH_SIZE = configs.BATCH_SIZE

    num_item = len(get_min_max_scores()[id])
    overall_min, overall_max = get_overall_score_range()[id]
    overall_range = overall_max - overall_min + 1
    analytic_min, analytic_max = get_analytic_score_range()[id]
    analytic_range = analytic_max - analytic_min + 1

    ######################
    # Cross Validation
    ######################
    cv_qwk = []
    cv_lwk = []
    cv_mae = []
    cv_rmse = []
    cv_corr = []
    for fold in range(Fold):
        #################
        # Reading Dataset
        #################
        train_path = data_path + '{}/fold-{}/train.pkl'.format(id, fold)
        test_path = data_path + '{}/fold-{}/test.pkl'.format(id, fold)

        read_configs = {
        'train_path': train_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
        }

        if word:
            word_vocab = read_word_vocab(read_configs)
            train_data, test_data = read_essays_words_cv(read_configs, word_vocab)

            embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
            embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
            embed_table = [embedd_matrix]
        else:
            pos_vocab = read_pos_vocab(read_configs)
            train_data, test_data = read_essays_pos_cv(read_configs, pos_vocab)

        max_sentlen = max(train_data['max_sentlen'], test_data['max_sentlen'])
        max_sentnum = max(train_data['max_sentnum'], test_data['max_sentnum'])

        if word:
            X_train = pad_hierarchical_text_sequences(train_data['words'], max_sentnum, max_sentlen)
            X_test = pad_hierarchical_text_sequences(test_data['words'], max_sentnum, max_sentlen)
        else:
            X_train = pad_hierarchical_text_sequences(train_data['pos_x'], max_sentnum, max_sentlen)
            X_test = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

        X_train_linguistic_features = np.array(train_data['features_x'])
        X_test_linguistic_features = np.array(test_data['features_x'])

        X_train_readability = np.array(train_data['readability_x'])
        X_test_readability = np.array(test_data['readability_x'])

        item_mask = get_attribute_mask_vector(id)
        Y_train_org = np.array(train_data['data_y'])[:, item_mask]
        Y_test_org = np.array(test_data['data_y'])[:, item_mask]

        # Overall Score
        train_data['y_scaled'] = get_scaled_down_scores(train_data['data_y'], train_data['prompt_ids'])
        Y_train_overall = np.array(train_data['y_scaled'])[:, item_mask]
        Y_train_overall = Y_train_overall[:, 0]

        # Analytic Score
        min_score = np.array(get_min_max_score_vector()[id]['min'])[item_mask]
        train_list = []
        for y in Y_train_org:
            y = to_categorical(y[1:] - min_score[1:], analytic_range)
            train_list.append(y)
        Y_train_analytic = np.array(train_list)

        train_features_list = [X_train, X_train_linguistic_features, X_train_readability]
        test_features_list = [X_test, X_test_linguistic_features, X_test_readability]
        
        #################
        # Define Model
        #################
        if word:
            if with_attn:
                model = build_Proposed_2way(len(word_vocab), max_sentnum, max_sentlen, X_test_readability.shape[1], X_train_linguistic_features.shape[1],
                                            configs, num_item, dim, analytic_range, embed_table)
            else:
                model = build_Proposed_2way_noatt(len(word_vocab), max_sentnum, max_sentlen, X_test_readability.shape[1], X_train_linguistic_features.shape[1],
                                                    configs, num_item, dim, analytic_range, embed_table)
        else:
            if with_attn:
                model = build_Proposed_2way(len(pos_vocab), max_sentnum, max_sentlen, X_test_readability.shape[1], X_train_linguistic_features.shape[1],
                                            configs, num_item, dim, analytic_range)
            else:
                model = build_Proposed_2way_noatt(len(pos_vocab), max_sentnum, max_sentlen, X_test_readability.shape[1], X_train_linguistic_features.shape[1],
                                                    configs, num_item, dim, analytic_range)

        ###############
        # Training model
        ###############
        eval = evaluator(num_item, overall_range, item_mask, analytic_range, test_features_list, id)
        for epoch in range(EPOCHS):
            print('Prompt ID: {}, SEED: {}'.format(id, seed))
            print('Attention?: {}'.format(with_attn))
            print('Word?: {}'.format(word))
            print('Dimension: {}, EPOCHS: {} / {}'.format(dim, epoch+1, EPOCHS))
            model.fit(x=train_features_list, y=[Y_train_overall, Y_train_analytic], batch_size=BATCH_SIZE, epochs=1)
            eval.evaluate_from_2way(model, Y_test_org, overall_range, min_score)
            eval.print_results()

        cv_qwk.append(eval.qwk)
        cv_lwk.append(eval.lwk)
        cv_rmse.append(eval.rmse)
        cv_mae.append(eval.mae)
        cv_corr.append(eval.corr)

        # print IRT parameter
        alpha = np.round(model.get_layer('mgpcm_layer').get_weights()[0], 3)
        beta = np.round(model.get_layer('mgpcm_layer').get_weights()[1]*-1, 2)
        print('Estimated alpha: {}'.format(alpha))
        print('Estimated beta: {}'.format(beta))

    ####################
    # Print final info
    ####################
    cv_mean_qwk = np.mean(np.array(cv_qwk), axis=0)
    cv_mean_lwk = np.mean(np.array(cv_lwk), axis=0)
    cv_mean_rmse = np.mean(np.array(cv_rmse), axis=0)
    cv_mean_mae = np.mean(np.array(cv_mae), axis=0)
    cv_mean_corr = np.mean(np.array(cv_corr), axis=0)
    print('-' * 100)
    print('Final info')
    print(' TEST_QWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(cv_mean_qwk), np.round(cv_mean_qwk, 3)))
    print(' TEST_LWK:  mean -> {:.3f}, each item -> {}'.format(np.mean(cv_mean_lwk), np.round(cv_mean_lwk, 3)))
    print(' TEST_RMSE: mean -> {:.3f}, each item -> {}'.format(np.mean(cv_mean_rmse), np.round(cv_mean_rmse, 3)))
    print(' TEST_MAE:  mean -> {:.3f}, each item -> {}'.format(np.mean(cv_mean_mae), np.round(cv_mean_mae, 3)))
    print(' TEST_CORR: mean -> {:.3f}, each item -> {}'.format(np.mean(cv_mean_corr), np.round(cv_mean_corr, 3)))

    ###################
    # Save outputs
    ###################
    if word:
        if with_attn:
            output_path = 'outputs/Proposed_2way/{}/word/'.format(seed)
        else:
            output_path = 'outputs/Proposed_2way_noatt/{}/word/'.format(seed)
    else:
        if with_attn:
            output_path = 'outputs/Proposed_2way/{}/pos/'.format(seed)
        else:
            output_path = 'outputs/Proposed_2way_noatt/{}/pos/'.format(seed)
    os.makedirs(output_path, exist_ok=True)
    print('Saving to {}'.format(output_path))
    pd.DataFrame(cv_mean_qwk).to_csv(output_path + 'qwk{}{}.csv'.format(id, dim), header=None, index=None)
    pd.DataFrame(cv_mean_lwk).to_csv(output_path + 'lwk{}{}.csv'.format(id, dim), header=None, index=None)
    pd.DataFrame(cv_mean_rmse).to_csv(output_path + 'rmse{}{}.csv'.format(id, dim), header=None, index=None)
    pd.DataFrame(cv_mean_mae).to_csv(output_path + 'mae{}{}.csv'.format(id, dim), header=None, index=None)
    pd.DataFrame(cv_mean_corr).to_csv(output_path + 'corr{}{}.csv'.format(id, dim), header=None, index=None)

if __name__=='__main__':
    main()