# import modules
import numpy as np
from statistics import mean
from sklearn import metrics

class ErrorCalculator:
    def __init__(self, train_target, train_prediction, model, test_target, test_prediction):
        self.train_target = train_target
        self.train_prediction = train_prediction
        self.model = model
        self.test_target = test_target
        self.test_prediction = test_prediction
        
    def get_residuals(self):
        residuals = self.model.resid
        min_resid = round(min(residuals), 2)
        max_resid = round(max(residuals), 2)
        mean_resid = round(mean(residuals), 2)
        
        return '{} {} {} {} {} {} {}'.format('RESIDUALS-->', 'MIN_resid: ', min_resid, 'MAX_resid: ', max_resid, 
                                          'MEAN_resid: ', mean_resid)
    
    def get_standardised_residuals(self):
        residuals = self.model.resid
        resid_std = np.std(residuals)
        standardised_resid = residuals/resid_std
        
        min_std_resid = round(min(standardised_resid), 2)
        max_std_resid = round(max(standardised_resid), 2)
        mean_std_resid = round(mean(standardised_resid), 2)
        
        return '{} {} {} {} {} {} {}'.format('STD RESIDUALS-->', 'MIN_std_resid:', min_std_resid, 'MAX_std_resid:', max_std_resid, 
                                          'MEAN_std_resid:', mean_std_resid)
    
    def get_mse(self):
        mse = round(metrics.mean_squared_error(self.train_target, self.train_prediction), 2)
        mse_test = round(metrics.mean_squared_error(self.test_target, self.test_prediction), 2)
        return '{} {} {} {} {}'.format('MSE-->', 'MSE_train:', mse, 'MSE_test:', mse_test)
    
    def get_rmse(self):
        mse = round(metrics.mean_squared_error(self.train_target, self.train_prediction), 2)
        mse_test = round(metrics.mean_squared_error(self.test_target, self.test_prediction), 2)
        rmse = round(np.sqrt(mse), 2)
        rmse_test = round(np.sqrt(mse_test), 2)
        return '{} {} {} {} {}'.format('RMSE-->', 'RMSE_train:', rmse, 'RMSE_test:', rmse_test)
    
    def error_summary(self):
        return '{}\n \n{}\n \n{}\n \n{}'.format(ErrorCalculator.get_residuals(self), 
                                          ErrorCalculator.get_standardised_residuals(self), 
                                          ErrorCalculator.get_mse(self), ErrorCalculator.get_rmse(self))