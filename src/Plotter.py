# import modules
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    
    def __init__(self, train_target, train_prediction, model):
        self.train_target = train_target
        self.train_prediction = train_prediction
        self.model = model

    def plot(self):
        residuals = self.model.resid
        _ = plt.hist(residuals)
        _ = plt.title("Residuals Histogram")
        _ = plt.xlabel("Residuals")
        return plt.show()

class HistogramPlotter(Plotter):
    pass


class ScatterPlotter(Plotter):
    
    def __init__(self, train_target, train_prediction, model, dataset):
        Plotter.__init__(self, train_target, train_prediction, model)
        self.dataset = dataset
    
    def plot(self):
        residuals = self.model.resid
        
        #Residuals vs Predicted Values
        plt.subplot(211)
        sns.residplot(self.train_prediction, residuals, lowess=True, data=self.dataset, 
             scatter_kws={'alpha':0.5},
             line_kws={'color':'red', 'lw':1, 'alpha':0.8})
        _ = plt.title("Residuals vs Predicted Values")
        _ = plt.xlabel("Predicted")
        _ = plt.ylabel("Residuals")
        plt.show()
        
        #Predicted Values vs Observed Values
        plt.subplot(212)
        sns.residplot(self.train_prediction, self.train_target, lowess=True, data=self.dataset, 
             scatter_kws={'alpha':0.5},
             line_kws={'color':'red', 'lw':1, 'alpha':0.8})
        _ = plt.title("Predicted Values vs Observed Values")
        _ = plt.xlabel("Predicted")
        _ = plt.ylabel("Residuals")
        plt.show()
