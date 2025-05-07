covid_df = load_and_preprocess_data()
data_with_features = create_prediction_features(covid_df)

november_start = pd.Timestamp('2020-11-01')
training_subset = data_with_features[data_with_features['date'] < november_start]
fitted_model, feature_normalizer, selected_features = train_prediction_model(training_subset)

evaluation_start = pd.Timestamp('2020-11-01')
evaluation_end = pd.Timestamp('2020-11-30')

performance_metrics = evaluate_predictions(
    covid_df, 
    fitted_model, 
    feature_normalizer, 
    selected_features, 
    evaluation_start, 
    evaluation_end
)

for idx, entry in performance_metrics.iterrows():
    date_str = entry['date'].strftime('%b %d')
    aci_cov = f"{entry['aci_coverage']:.3f}"
    aci_w = f"{entry['aci_width']:.1f}"
    qt_cov = f"{entry['qt_coverage']:.3f}"
    qt_w = f"{entry['qt_width']:.1f}"
    print(f"| {date_str:^8} | {aci_cov:^12} | {aci_w:^14} | {qt_cov:^11} | {qt_w:^14} |")

print(f"• ACI Method: {performance_metrics['aci_coverage'].mean():.3f} coverage with {performance_metrics['aci_width'].mean():.1f} width")
print(f"• QT Method:  {performance_metrics['qt_coverage'].mean():.3f} coverage with {performance_metrics['qt_width'].mean():.1f} width")

if len(performance_metrics) < 30:
    print(f"\n Warning: Analysis includes only {len(performance_metrics)} days of data")

def visualize_results(metrics_df):
    if not pd.api.types.is_datetime64_dtype(metrics_df['date']):
        metrics_df['date'] = pd.to_datetime(metrics_df['date'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(metrics_df['date'], metrics_df['aci_coverage'], 'blue', linewidth=2, label='ACI')
    axes[0, 0].plot(metrics_df['date'], metrics_df['qt_coverage'], 'red', linewidth=2, label='QT')
    axes[0, 0].axhline(y=0.9, color='black', linestyle='--', alpha=0.7, label='Target (90%)')
    axes[0, 0].set_title('Coverage Rate Analysis')
    axes[0, 0].legend()
    
    width_data = metrics_df[['date', 'aci_width', 'qt_width']].groupby(pd.Grouper(key='date', freq='W-MON')).mean()
    x_pos = np.arange(len(width_data))
    axes[0, 1].bar(x_pos - 0.35/2, width_data['aci_width'], 0.35, label='ACI', color='blue', alpha=0.7)
    axes[0, 1].bar(x_pos + 0.35/2, width_data['qt_width'], 0.35, label='QT', color='red', alpha=0.7)
    axes[0, 1].set_title('Weekly Average Interval Width')
    axes[0, 1].legend()
    
    axes[1, 0].scatter(metrics_df['avg_pred'], metrics_df['avg_true'], alpha=0.7, c='purple')
    max_val = max(metrics_df['avg_pred'].max(), metrics_df['avg_true'].max())
    axes[1, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    z = np.polyfit(metrics_df['avg_pred'], metrics_df['avg_true'], 1)
    axes[1, 0].plot(metrics_df['avg_pred'], np.poly1d(z)(metrics_df['avg_pred']), "r-", alpha=0.7)
    axes[1, 0].set_title('Prediction vs Actual')
    
    data_to_plot = metrics_df[['date', 'tau_aci_avg', 'tau_qt_avg']].set_index('date')
    X, Y = np.meshgrid(np.arange(2), np.arange(len(data_to_plot)))
    Z = np.array([data_to_plot['tau_aci_avg'].values, data_to_plot['tau_qt_avg'].values]).T
    c = axes[1, 1].pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    axes[1, 1].set_title('Tau Parameter Evolution')
    fig.colorbar(c, ax=axes[1, 1], label='Tau Value')
    
    plt.tight_layout()
    plt.show()  
    plt.close(fig)  

visualize_results(performance_metrics)

