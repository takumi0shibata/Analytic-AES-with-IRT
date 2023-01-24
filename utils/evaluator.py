import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from utils.general_utils import get_min_max_score_vector


class evaluator():
    def __init__(self, num_item, overall_range, item_mask, analytic_range, test_features_list, prompt_id):
        self.prompt_id = prompt_id
        self.item_mask = item_mask
        self.overall_range = overall_range
        self.analytic_range = analytic_range
        self.num_item = num_item
        self.test_features_list = test_features_list
        self.qwk = None
        self.lwk = None
        self.rmse = None
        self.mae = None
        self.corr = None


    def calc_qwk(self, y_true, y_pred):
        qwk_scores = []
        for i in range(self.num_item):
            if i == 0:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='quadratic', labels=[i for i in range(self.overall_range)])
                qwk_scores.append(kappa_score)
            else:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='quadratic', labels=[i for i in range(self.analytic_range)])
                qwk_scores.append(kappa_score)
        self.qwk = np.array(qwk_scores)


    def calc_lwk(self, y_true, y_pred):
        lwk_scores = []
        for i in range(self.num_item):
            if i == 0:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='linear', labels=[i for i in range(self.overall_range)])
                lwk_scores.append(kappa_score)
            else:
                kappa_score = cohen_kappa_score(y_true[:, i], y_pred[:, i], weights='linear', labels=[i for i in range(self.analytic_range)])
                lwk_scores.append(kappa_score)
        self.lwk = np.array(lwk_scores)


    def calc_rmse(self, y_true, y_pred):
        rmse_scores = []
        for i in range(self.num_item):
            rmse_score = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
            rmse_scores.append(rmse_score)
        self.rmse = np.array(rmse_scores)


    def calc_mae(self, y_true, y_pred):
        mae_scores = []
        for i in range(self.num_item):
            mae_score = mean_absolute_error(y_true[:, i], y_pred[:, i])
            mae_scores.append(mae_score)
        self.mae = np.array(mae_scores)


    def calc_corr(self, y_true, y_pred):
        corr_scores = []
        for i in range(self.num_item):
            corr_score = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
            corr_scores.append(corr_score)
        self.corr = np.array(corr_scores)


    def evaluate_from_reg(self, model, y_true_org):
        # Calculate score range
        min_score = np.array(get_min_max_score_vector()[self.prompt_id]['min'])[self.item_mask]
        max_score = np.array(get_min_max_score_vector()[self.prompt_id]['max'])[self.item_mask]
        score_range = max_score - min_score

        # Predict scores
        y_pred = model.predict(self.test_features_list)
        y_pred = y_pred * score_range
        y_pred = np.round(y_pred)

        # Set the minimum score to 0
        y_true = y_true_org - min_score

        # Calculate metrics
        self.calc_qwk(y_true, y_pred)
        self.calc_lwk(y_true, y_pred)
        self.calc_rmse(y_true, y_pred)
        self.calc_mae(y_true, y_pred)
        self.calc_corr(y_true, y_pred)


    def evaluate_from_prob(self, model, y_true_org, min_score, predict_option='ex'):
        # Predict scores
        y_pred = model.predict(self.test_features_list)
        if predict_option == 'ex':
            y_pred = np.sum(y_pred * np.arange(0, self.overall_range), axis=-1) # expected score
            y_pred = np.round(y_pred)
        elif predict_option == 'argmax':
            y_pred = np.argmax(y_pred, axis=-1)

        # Set the minimum score to 0
        y_true = y_true_org - min_score

        # Calculate metrics
        self.calc_qwk(y_true, y_pred)
        self.calc_lwk(y_true, y_pred)
        self.calc_rmse(y_true, y_pred)
        self.calc_mae(y_true, y_pred)
        self.calc_corr(y_true, y_pred)


    def evaluate_from_2way(self, model, y_true_org, overall_range, min_score, pred_opt='ex'):
        # Predict Scores
        y_pred_overall, y_pred_anaytic = model.predict(self.test_features_list)

        # Rescale overall score
        y_pred_overall = y_pred_overall * (overall_range - 1)
        y_pred_overall = np.round(y_pred_overall)

        # Rescale analytic scores
        if pred_opt == 'ex':
            y_pred_anaytic = np.sum(y_pred_anaytic * np.arange(0, self.analytic_range), axis=-1)
            y_pred_anaytic = np.round(y_pred_anaytic)
        elif pred_opt == 'argmax':
            y_pred_anaytic = np.argmax(y_pred, axis=-1)

        # Compiled scores
        y_pred = np.concatenate([y_pred_overall, y_pred_anaytic], axis=1)

        # Set the minimum score to 0
        y_true = y_true_org - min_score

        # Calculate metrics
        self.calc_qwk(y_true, y_pred)
        self.calc_lwk(y_true, y_pred)
        self.calc_rmse(y_true, y_pred)
        self.calc_mae(y_true, y_pred)
        self.calc_corr(y_true, y_pred)


    def print_results(self):
        print('TEST_QWK:  Mean -> {:.3f}, Each item -> {}'.format(np.mean(self.qwk), np.round(self.qwk, 3)))
        print('TEST_LWK:  Mean -> {:.3f}, Each item -> {}'.format(np.mean(self.lwk), np.round(self.lwk, 3)))
        print('TEST_RMSE: Mean -> {:.3f}, Each item -> {}'.format(np.mean(self.rmse), np.round(self.rmse, 3)))
        print('TEST_MAE:  Mean -> {:.3f}, Each item -> {}'.format(np.mean(self.mae), np.round(self.mae, 3)))
        print('TEST_CORR: Mean -> {:.3f}, Each item -> {}'.format(np.mean(self.corr), np.round(self.corr, 3)))
        print('-' * 100)
