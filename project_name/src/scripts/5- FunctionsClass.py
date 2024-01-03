import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import *
import random

class DataVisualizer:
    def __init__(self, train_data_path='normalized_train_data.csv',
                 cleaned_train_data_path='cleaned_train_data.csv',
                 test_data_path='normalized_test_data.csv'):
        self.train_data = pd.read_csv(train_data_path)
        self.train_data_c = pd.read_csv(cleaned_train_data_path)
        self.test_data = pd.read_csv(test_data_path)
        self.colors = ['blue', 'Orange', 'green', 'Purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Creates histograms of the output and of selected inputs
    # Enter number of variables ---> 2 yields input2
    def hist_all(self, x=[2,3,4]):
        # Creating a 22x2 figure for 44 scatter plots
        fig, axs = plt.subplots(len(x)+1, 2, figsize=(20, 80))
        # Adjusting margins between subplots
        fig.subplots_adjust(hspace=0.4, wspace=0.2)
        
        for i, j in enumerate(x):
            # plot befor cleaning
            axs[i, 0].hist(self.train_data['input%d'%j], bins=20,
                              color='green', edgecolor='black')
            axs[i, 0].set_xlabel('Normalized data')
            axs[i, 0].set_ylabel('Frequency')
            axs[i, 0].set_title('input%d-original'%j)
            # plot after cleaning
            axs[i, 1].hist(self.train_data_c['input%d'%j], bins=20,
                              color='pink', edgecolor='black')
            axs[i, 1].set_xlabel('Normalized data')
            axs[i, 1].set_ylabel('Frequency')
            axs[i, 1].set_title('input%d-cleaned'%j)
        # add the output befor cleaning
        axs[i+1, 0].hist(self.train_data['output'], bins=20,
                          color='green', edgecolor='black')
        axs[i+1, 0].set_xlabel('Normalized data')
        axs[i+1, 0].set_ylabel('Frequency')
        axs[i+1, 0].set_title('output-original')

        # add the output after cleaning
        axs[i+1, 1].hist(self.train_data_c['output'], bins=20,
                          color='pink', edgecolor='black')
        axs[i+1, 1].set_xlabel('Normalized data')
        axs[i+1, 1].set_ylabel('Frequency')
        axs[i+1, 1].set_title('output-cleaned')
        
        # Save the plot
##        plt.savefig('Histograms.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        pass
    
## ----------------------------------------------------------------

    # Creates scatter plots of the output vs. any input dimension
    # Enter number of variables ---> 2 yields input2
    def scatterplt(self, x=[2,3,4]):
        y_train = self.train_data['output']        
        nn = ceil(sqrt(len(x)))
        if len(x) == 1:
            nrows = 2
        else:
            nrows = nn
        ncols = nn

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adding horizontal space between subplots

        # Create plots  
        for i, j in enumerate(x):
            plt.subplot(nrows, ncols, i+1)
            plt.scatter(self.train_data[f'input{j}'], y_train,
                        color=random.choice(self.colors))
            plt.xlabel(f'input{j}')
            plt.ylabel('output')

        # Hide extra subplots if not used
        for ax in axs.flatten()[len(x):]:
            ax.axis('off')
        # Save the plot    
##        plt.savefig(f'Scatter plot {x}.png', dpi=300, bbox_inches='tight')

        plt.show()
        
        pass


# To call the class:
# 1. from FunctionsClass import DataVisualizer
# 2. data_visualizer = DataVisualizer() 
# 3. data_visualizer.outlier_det(range(2,6))


