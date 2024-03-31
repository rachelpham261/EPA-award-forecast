import json
import matplotlib.pyplot as plt

# Load data from JSON files
with open('dict_best_params_ridge.json', 'r') as file:
    ridge_data = json.load(file)

with open('dict_best_params_xgb.json', 'r') as file:
    xgb_data = json.load(file)

# Extract the number of features, RMSE, and R2 from the data
features = sorted(map(int, ridge_data.keys()))
ridge_rmse = [ridge_data[str(f)]["rmse"] for f in features]
xgb_rmse = [xgb_data[str(f)]["rmse"] for f in features]
ridge_r2 = [ridge_data[str(f)]["r2"] for f in features]
xgb_r2 = [xgb_data[str(f)]["r2"] for f in features]

# Simplify the feature list for clarity in the plot (choose every 10th feature)
feature_steps = 10
simplified_features = features[::feature_steps]
simplified_ridge_rmse = [ridge_rmse[i] for i in range(0, len(features), feature_steps)]
simplified_xgb_rmse = [xgb_rmse[i] for i in range(0, len(features), feature_steps)]
simplified_ridge_r2 = [ridge_r2[i] for i in range(0, len(features), feature_steps)]
simplified_xgb_r2 = [xgb_r2[i] for i in range(0, len(features), feature_steps)]

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# RMSE plot
axs[0].plot(simplified_features, simplified_ridge_rmse, label='Ridge', marker='o', color='tab:blue')
axs[0].plot(simplified_features, simplified_xgb_rmse, label='XGBoost', marker='s', color='tab:orange')
axs[0].set_xlabel('Number of Features')
axs[0].set_ylabel('RMSE')
axs[0].set_title('RMSE Comparison')
axs[0].grid(True)

# R2 plot
axs[1].plot(simplified_features, simplified_ridge_r2, marker='o', color='tab:blue')
axs[1].plot(simplified_features, simplified_xgb_r2, marker='s', color='tab:orange')
axs[1].set_xlabel('Number of Features')
axs[1].set_ylabel('R2 Score')
axs[1].set_title('R2 Comparison')
axs[1].grid(True)

# Adjusting legend
fig.legend(['Ridge', 'XGBoost'], loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Save the figure to a file
plt.savefig('comparison_figure.png', bbox_inches='tight', dpi=300)

