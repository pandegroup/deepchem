import numpy as np
import deepchem as dc
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from future import print_function, division, unicode_literals

from bd13_datasets import aa_charset, load_bd2013_human
from deepmhc import DeepMHC

def create_model(num_amino_acids, pad_len, dropout_p=0.5, l2_reg=0.001):
    """
    Create DeepMHC model with regularization and dropout
    
    Args:
        num_amino_acids (int): Number of unique amino acids
        pad_len (int): Sequence padding length
        dropout_p (float): Dropout probability
        l2_reg (float): L2 regularization strength
    
    Returns:
        DeepMHC model with configured parameters
    """
    model = DeepMHC(
        num_amino_acids=num_amino_acids, 
        pad_length=pad_len,
        dropout_rate=dropout_p,
        l2_reg=l2_reg
    )
    return model

def train_with_cross_validation(model, datasets, nb_epochs=200, k_folds=5):
    """
    Perform k-fold cross-validation
    
    Args:
        model (DeepMHC): Initialized model
        datasets (list): Training and test datasets
        nb_epochs (int): Maximum training epochs
        k_folds (int): Number of cross-validation folds
    
    Returns:
        List of performance scores for each fold
    """
    train_dataset, test_dataset = datasets
    metric = dc.metrics.Metric(metric=dc.metrics.pearsonr, mode="regression")
    
    # Callbacks for adaptive training
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=0.0001
    )
    
    performance_scores = []
    
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_index, val_index) in enumerate(kfold.split(train_dataset.X), 1):
        print(f"Training Fold {fold}")
        
        # Split data
        train_subset = train_dataset.select(train_index)
        val_subset = train_dataset.select(val_index)
        
        # Reset and retrain model for each fold
        current_model = create_model(num_amino_acids, pad_len)
        
        # Train with early stopping and learning rate scheduling
        current_model.fit(
            train_subset, 
            nb_epoch=nb_epochs, 
            validation_data=val_subset,
            callbacks=[early_stop, lr_scheduler]
        )
        
        # Evaluate model
        train_scores = current_model.evaluate(train_subset, [metric])
        test_scores = current_model.evaluate(val_subset, [metric])
        
        performance_scores.append({
            'train_pearsonr': train_scores['pearsonr'],
            'val_pearsonr': test_scores['pearsonr']
        })
        
        print(f"Fold {fold} Train Pearson: {train_scores['pearsonr']}")
        print(f"Fold {fold} Validation Pearson: {test_scores['pearsonr']}\n")
    
    return performance_scores

def main():
    # Configuration
    pad_len = 13
    num_amino_acids = len(aa_charset)
    mhc_allele = "HLA-A*02:01"
    dropout_p = 0.5
    batch_size = 64
    nb_epochs = 200

    # Load dataset
    tasks, datasets, transformers = load_bd2013_human(
        mhc_allele=mhc_allele, 
        seq_len=9, 
        pad_len=pad_len
    )

    # Create and train model
    model = create_model(num_amino_acids, pad_len, dropout_p)
    cross_val_results = train_with_cross_validation(
        model, datasets, nb_epochs=nb_epochs
    )

    # Summarize cross-validation results
    avg_train_pearsonr = np.mean([
        result['train_pearsonr'] for result in cross_val_results
    ])
    avg_val_pearsonr = np.mean([
        result['val_pearsonr'] for result in cross_val_results
    ])

    print("Cross-Validation Summary:")
    print(f"Average Train Pearson R: {avg_train_pearsonr}")
    print(f"Average Validation Pearson R: {avg_val_pearsonr}")

if __name__ == "__main__":
    main()
