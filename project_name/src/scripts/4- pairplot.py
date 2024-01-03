import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd


# ------------------------- load the data --------------------------
train_data = pd.read_csv('normalized_train_data.csv')


# ----- Pair plot for exploring relationships between variables ----
sns.pairplot(train_data, hue="output", diag_kind="kde",
             vars=train_data.columns)
plt.title('Pairplot of Cleaned Dataset')
plt.savefig('Pairplot_cleaned.png', dpi=300, bbox_inches='tight')
plt.show()






