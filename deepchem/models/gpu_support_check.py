import deepchem as dc
import tensorflow as tf

# Check if TensorFlow is using GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available. DeepChem will use GPU for computations.")
else:
    print("GPU is not available. Using CPU instead.")

# Example: Load a sample dataset and use a simple model
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

# Create a model
model = dc.models.GraphConvModel(n_tasks=len(tasks), mode='regression', use_queue=False)

# Train the model
model.fit(train_dataset, nb_epoch=10)

# Evaluate the model
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print("Validation set score:", model.evaluate(valid_dataset, [metric], transformers))

