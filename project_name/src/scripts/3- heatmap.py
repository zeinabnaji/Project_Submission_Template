import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------- load the data ---------------

train_data = pd.read_csv('normalized_train_data.csv')
train_data_c = pd.read_csv('cleaned_train_data.csv')


# --------------------- Correlation Heatmap--------------------
# ------- Original dataset
correlation_matrix = train_data.corr() #train_data.iloc[:, :5].corr()
ax=sns.heatmap(correlation_matrix, cmap="PiYG", annot=True,
            annot_kws={"fontsize": 4}, vmin=-1.00, vmax=1.00)

# Modify font sizes for x-axis and y-axis labels
plt.xticks(fontsize=5)  
plt.yticks(fontsize=5)

# Colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=5)

plt.title('Correlation Heatmap - Original Dataset', fontsize=8)
plt.tight_layout()
plt.savefig('Correlation_Heatmap_Original.png', dpi=300)
plt.show()

# ------- Cleaned dataset
correlation_matrix = train_data_c.corr()
ax=sns.heatmap(correlation_matrix, cmap="PiYG", annot=True,
            annot_kws={"fontsize": 4}, vmin=-1.00, vmax=1.00)

# Modify font sizes for x-axis and y-axis labels
plt.xticks(fontsize=5)  
plt.yticks(fontsize=5)

# Colorbar font size
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=5)

plt.title('Correlation Heatmap - Cleaned Dataset', fontsize=8)
plt.tight_layout()
plt.savefig('Correlation_Heatmap_Cleaned.png', dpi=300)
plt.show()
