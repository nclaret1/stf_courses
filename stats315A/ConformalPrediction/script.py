from conformal_multiclass import *
from conformal_cifar import *
from cifar_processing import *
import pandas as pd 
import numpy as np


train_data = load_numpy_into_data('cifar100_train_features.npy', 'cifar100_train_labels.npy')
test_data = load_numpy_into_data('cifar100_test_features.npy', 'cifar100_test_labels.npy')

experiments = 5
projection_dim = 10
feature_dim = 2048


results = {
    'static_coverage': [], 'static_sizes': [],
    'conditional_coverage': [], 'conditional_sizes': [],
    'marginal_coverage_static': [], 'marginal_sizes_static': [],
    'marginal_coverage_cond': [], 'marginal_sizes_cond': []
}

for exp in range(experiments):
    print(f"\nExperiment {exp + 1}/{experiments}")
    
    print("Sub-quesgtion (i)")
    train_subset, val_subset = split_into_train_and_validation(train_data, 0.5)
    model = train_prediction_model(train_subset, num_epochs=5)
    
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    val_features, val_logits, val_labels = load_data_to_numpy(val_loader, model)
    
    print("Sub-question (ii): static threshold")
    nonconformity_scores = np.array([val_logits[i, val_labels[i]] for i in range(len(val_labels))])
    static_tau = find_static_threshold(nonconformity_scores)
    
    print("static threshold")
    W = np.random.randn(projection_dim, feature_dim)
    projected_val_features = val_features @ W.T
    conditional_theta = find_conditional_threshold(nonconformity_scores, projected_val_features)
    
    print("Sub-quesgtion (iii)")
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    test_features, test_logits, test_labels = load_data_to_numpy(test_loader, model)
    
    projected_test_features = test_features @ W.T
    
    static_sets = create_prediction_sets(test_logits, static_tau)
    
    X_test = np.column_stack([np.ones(len(projected_test_features)), projected_test_features])
    conditional_thresholds = X_test @ conditional_theta
    conditional_sets = [np.where(test_logits[i] >= conditional_thresholds[i])[0] 
                        for i in range(len(test_logits))]
    
    static_coverage, static_size = check_coverage(static_sets, test_labels)
    conditional_coverage, conditional_size = check_coverage(conditional_sets, test_labels)
    
    results['marginal_coverage_static'].append(static_coverage)
    results['marginal_sizes_static'].append(static_size)
    results['marginal_coverage_cond'].append(conditional_coverage)
    results['marginal_sizes_cond'].append(conditional_size)
    
    for dim in range(projection_dim):
        proj_values = projected_test_features[:, dim]
        
        low_quantile = np.quantile(proj_values, 0.1)
        high_quantile = np.quantile(proj_values, 0.9)
        
        extreme_indices = np.where((proj_values <= low_quantile) | (proj_values >= high_quantile))[0]
        
        extreme_labels = test_labels[extreme_indices]
        extreme_static_sets = [static_sets[i] for i in extreme_indices]
        extreme_conditional_sets = [conditional_sets[i] for i in extreme_indices]
        
        ext_static_cov, ext_static_size = check_coverage(extreme_static_sets, extreme_labels)
        ext_cond_cov, ext_cond_size = check_coverage(extreme_conditional_sets, extreme_labels)
        
        results['static_coverage'].append(ext_static_cov)
        results['static_sizes'].append(ext_static_size)
        results['conditional_coverage'].append(ext_cond_cov)
        results['conditional_sizes'].append(ext_cond_size)



def visualize_results(results):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    
    axes[0, 0].hist(results['static_coverage'], bins=20, color='blue', alpha=0.7, label='Static')
    axes[0, 0].hist(results['conditional_coverage'], bins=20, color='orange', alpha=0.7, label='Conditional')
    axes[0, 0].set_title('Coverage Distribution')
    axes[0, 0].set_xlabel('Coverage')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    axes[0, 1].violinplot([results['static_sizes'], results['conditional_sizes']], showmeans=True)
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(['Static', 'Conditional'])
    axes[0, 1].set_title('Set Size Distribution')
    axes[0, 1].set_ylabel('Set Size')
    
    axes[1, 0].scatter(results['static_coverage'], results['static_sizes'], color='blue', alpha=0.5, label='Static')
    axes[1, 0].scatter(results['conditional_coverage'], results['conditional_sizes'], color='orange', alpha=0.5, label='Conditional')
    axes[1, 0].set_title('Coverage vs. Set Size')
    axes[1, 0].set_xlabel('Coverage')
    axes[1, 0].set_ylabel('Set Size')
    axes[1, 0].legend()
    
    axes[1, 1].boxplot([results['static_sizes'], results['conditional_sizes']], labels=['Static', 'Conditional'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[1, 1].set_title('Boxplot of Set Sizes')
    axes[1, 1].set_ylabel('Size')
    
    plt.tight_layout()
    plt.savefig('conformal_results_v2.png')
    plt.show()
    
    print("\nSummary of Experimental Results:")
    print(f"Static coverage: {np.mean(results['marginal_coverage_static']):.3f} ± {np.std(results['marginal_coverage_static']):.3f}")
    print(f"Static set size: {np.mean(results['marginal_sizes_static']):.3f} ± {np.std(results['marginal_sizes_static']):.3f}")
    print(f"Conditional coverage: {np.mean(results['marginal_coverage_cond']):.3f} ± {np.std(results['marginal_coverage_cond']):.3f}")
    print(f"Conditional set size: {np.mean(results['marginal_sizes_cond']):.3f} ± {np.std(results['marginal_sizes_cond']):.3f}")
