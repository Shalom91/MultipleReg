# import modules
import matplotlib.pyplot as plt
import numpy as np


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

