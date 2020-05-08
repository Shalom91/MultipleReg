# import modules
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Plotter:
    def __init__(self, train_target, train_prediction, model):
        self.train_target = train_target
        self.train_prediction = train_prediction
        self.model = model
        
    def plot_hist(self):
        residuals = self.model.resid
        resid_std = np.std(residuals)
        standardised_resid = residuals/resid_std
        
        plt.subplot(121)
        _ = plt.hist(residuals)
        _ = plt.title("Residuals Histogram", pad=10)
        _ = plt.xlabel("Residuals") 
        
        
        plt.subplot(122)
        plt.subplots_adjust(wspace=1)
        _ = plt.hist(standardised_resid)
        _  = plt.title("STD Residuals Histogram", pad=10)
        _ = plt.xlabel("Standardised Residuals") 
        
        
        plt.tight_layout()
        plt.show()

class ScatterPlotter(Plotter):
    
    def __init__(self, train_target, train_prediction, model, dataset):
        Plotter.__init__(self, train_target, train_prediction, model)
        self.dataset = dataset
    
    def resid_fitted_scatter(self):
        residuals = self.model.resid
        plt.subplot(211)
        sns.residplot(self.train_prediction, residuals, lowess=True, data=self.dataset, 
             scatter_kws={'alpha':0.5},
             line_kws={'color':'red', 'lw':1, 'alpha':0.8})
        _ = plt.title("Residuals vs Predicted Values")
        _ = plt.xlabel("Predicted")
        _ = plt.ylabel("Residuals")
        plt.show()
        
        plt.subplot(212)
        sns.residplot(self.train_prediction, self.train_target, lowess=True, data=self.dataset, 
             scatter_kws={'alpha':0.5},
             line_kws={'color':'red', 'lw':1, 'alpha':0.8})
        _ = plt.title("Predicted Values vs Observed Values")
        _ = plt.xlabel("Predicted")
        _ = plt.ylabel("Residuals")
        plt.show()