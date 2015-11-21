"""
Top level script to featurize input, train models, and evaluate them.
"""
import argparse
import gzip
import numpy as np
import cPickle as pickle
from deep_chem.utils.featurize import generate_directories
from deep_chem.utils.featurize import extract_data
from deep_chem.utils.featurize import generate_targets
from deep_chem.utils.featurize import generate_features
from deep_chem.utils.featurize import generate_smiles
from deep_chem.utils.featurize import generate_vs_utils_features
from deep_chem.models.standard import fit_singletask_models
from deep_chem.utils.load import get_target_names
from deep_chem.utils.load import process_datasets
from deep_chem.utils.load import transform_data
from deep_chem.utils.evaluate import results_to_csv
from deep_chem.utils.save import save_model
from deep_chem.utils.save import load_model
from deep_chem.utils.evaluate import compute_model_performance

def parse_args(input_args=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Modes')
 
  # FEATURIZE FLAGS
  featurize_cmd = subparsers.add_parser("featurize",
                      help="Featurize raw input data. The word 'endpoint' below is used\n"
                           "to refer to a field in the input (a column-name in a csv or xlsx)\n"
                           "or a field name in an sdf file or pandas dataframe.")
  featurize_cmd.add_argument("--input-file", required=1,
                      help="Input file with data.")
  featurize_cmd.add_argument("--input-type", default="csv",
                      choices=["xlsx", "csv", "pandas", "sdf"],
                      help="Type of input file. If pandas, input must be a pkl.gz\n"
                           "containing a pandas dataframe. If sdf, should be in\n"
                           "(perhaps gzipped) sdf file.")
  featurize_cmd.add_argument("--delimiter", default=",",
                      help="If csv input, delimiter to use for read csv file")
  featurize_cmd.add_argument("--fields", required=1, nargs="+",
                      help = "Names of fields.")
  featurize_cmd.add_argument("--field-types", required=1, nargs="+",
                      choices=["string", "float", "list-string", "list-float", "ndarray"],
                      help="Type of data in fields.")
  featurize_cmd.add_argument("--feature-endpoints", type=str, nargs="+",
                      help="Optional endpoint that holds pre-computed feature vector")
  featurize_cmd.add_argument("--prediction-endpoint", type=str, required=1,
                      help="Name of measured endpoint to predict.")
  featurize_cmd.add_argument("--split-endpoint", type=str, default=None,
                      help="Name of endpoint specifying train/test split.")
  featurize_cmd.add_argument("--smiles-endpoint", type=str, default="smiles",
                      help="Name of endpoint specifying SMILES for molecule.")
  featurize_cmd.add_argument("--id-endpoint", type=str, default=None,
                      help="Name of endpoint specifying unique identifier for molecule.\n"
                           "If none is specified, then smiles-endpoint is used as identifier.")
  # TODO(rbharath): This should be moved to train-tests-split
  featurize_cmd.add_argument("--threshold", type=float, default=None,
                      help="If specified, will be used to binarize real-valued prediction-endpoint.")
  featurize_cmd.add_argument("--name", required=1,
                      help="Name of the dataset.")
  featurize_cmd.add_argument("--out", required=1,
                      help="Folder to generate processed dataset in.")
  featurize_cmd.set_defaults(func=featurize_input)

  # Train/Test Splits flag 
  train_test_cmd = subparsers.add_parser("train-test-split",
                      help="Apply standard data transforms to raw features generated by featurize,\n"
                           "then split data into train/test and store data as (X,y) matrices.")
  train_test_cmd.add_argument("--input-transforms", nargs="+", default=[],
                      choices=["normalize-and-truncate"],
                      help="Transforms to apply to input data.")
  train_test_cmd.add_argument("--output-transforms", nargs="+", default=[],
                      choices=["log", "normalize"],
                      help="Transforms to apply to output data.")
  train_test_cmd.add_argument("--feature-types", nargs="+", required=1,
                      help="Types of featurizations to use.\n"
                           "Each featurization must correspond to subdirectory in\n"
                           "generated data directory.")
  train_test_cmd.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")
  train_test_cmd.add_argument("--splittype", type=str, default="scaffold",
                       choices=["scaffold", "random", "specified"],
                       help="Type of train/test data-splitting.\n"
                            "scaffold uses Bemis-Murcko scaffolds.\n"
                            "specified requires that split be in original data.")
  train_test_cmd.add_argument("--weight-positives", type=bool, default=False,
                  help="Weight positive examples to have same total weight as negatives.")
  train_test_cmd.add_argument("--mode", default="singletask",
                      choices=["singletask", "multitask"],
                      help="Type of model being built.")
  train_test_cmd.add_argument("--train-out", type=str, required=1,
                     help="Location to save train set.")
  train_test_cmd.add_argument("--test-out", type=str, required=1,
                     help="Location to save test set.")
  train_test_cmd.set_defaults(func=train_test_input)

  # TRAIN FLAGS
  train_cmd = subparsers.add_parser("fit",
                  help="Fit a model to training data.")
  group = train_cmd.add_argument_group("load-and-transform")
  group.add_argument("--task-type", default="classification",
                      choices=["classification", "regression"],
                      help="Type of learning task.")
  group.add_argument("--saved-data", required=1,
                     help="Location of saved transformed data.")
  # TODO(rbharath): CODE SMELL. This shouldn't be shuttled around
  group.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")

  group = train_cmd.add_argument_group("model")
  group.add_argument("--model", required=1,
                      choices=["logistic", "rf_classifier", "rf_regressor",
                      "linear", "ridge", "lasso", "lasso_lars", "elastic_net",
                      "singletask_deep_network", "multitask_deep_network",
                      "3D_cnn", "neural_fingerprint"],
                      help="Type of model to build. Some models may allow for\n"
                           "further specification of hyperparameters. See flags below.")

  group = train_cmd.add_argument_group("Neural Net Parameters")
  group.add_argument("--n-hidden", type=int, default=500,
                      help="Number of hidden neurons for NN models.")
  group.add_argument("--learning-rate", type=float, default=0.01,
                  help="Learning rate for NN models.")
  group.add_argument("--dropout", type=float, default=0.5,
                  help="Learning rate for NN models.")
  group.add_argument("--n-epochs", type=int, default=50,
                  help="Number of epochs for NN models.")
  group.add_argument("--batch-size", type=int, default=32,
                  help="Number of examples per minibatch for NN models.")
  group.add_argument("--decay", type=float, default=1e-4,
                  help="Learning rate decay for NN models.")
  group.add_argument("--validation-split", type=float, default=0.0,
                  help="Percent of training data to use for validation.")

  group = train_cmd.add_argument_group("save")
  group.add_argument("--saved-out", type=str, required=1,
                  help="Location to save trained model.")
  train_cmd.set_defaults(func=fit_model)

  eval_cmd = subparsers.add_parser("eval",
                help="Evaluate trained model on test data processed by transform.")
  group = eval_cmd.add_argument_group("load model/data")
  group.add_argument("--saved-model", type=str, required=1,
                  help="Location from which to load saved model.")
  group.add_argument("--saved-data", required=1,
                     help="Location of saved transformed data.")
  # TODO(rbharath): CODE SMELL. This shouldn't be shuttled around
  group.add_argument("--paths", nargs="+", required=1,
                      help="Paths to input datasets.")
  group.add_argument("--modeltype", required=1,
                      choices=["autograd", "sklearn", "keras-graph", "keras-sequential"],
                      help="Type of model to load.")
  # TODO(rbharath): This argument seems a bit extraneous. Is it really
  # necessary?
  group.add_argument("--task-type", default="classification",
                      choices=["classification", "regression"],
                      help="Type of learning task.")
  group = eval_cmd.add_argument_group("Classification metrics")
  group.add_argument("--compute-aucs", action="store_true", default=False,
                      help="Compute AUC for trained models on test set.")
  group.add_argument("--compute-accuracy", action="store_true", default=False,
                      help="Compute accuracy for trained models on test set.")
  group.add_argument("--compute-recall", action="store_true", default=False,
                      help="Compute recall for trained models on test set.")
  group.add_argument("--compute-matthews-corrcoef", action="store_true", default=False,
                      help="Compute Matthews Correlation Coefficient for trained models on test set.")

  group = eval_cmd.add_argument_group("Regression metrics")
  group.add_argument("--compute-r2s", action="store_true", default=False,
                     help="Compute R^2 for trained models on test set.")
  group.add_argument("--compute-rms", action="store_true", default=False,
                     help="Compute RMS for trained models on test set.")

  eval_cmd.add_argument("--csv-out", type=str, default=None,
                     help="Outputted predictions on the test set.")
  eval_cmd.set_defaults(func=eval_trained_model)

  return parser.parse_args(input_args)

# TODO(rbharath): This function needs to take feature-types as an argument
# rather than generating all features for all compounds.
def featurize_input(args):
  """Featurizes raw input data."""
  if len(args.fields) != len(args.field_types):
    raise ValueError("number of fields does not equal number of field types")
  if args.id_endpoint is None:
    args.id_endpoint = args.smiles_endpoint
  out_x_pkl, out_y_pkl = generate_directories(args.name, args.out, 
      args.feature_endpoints)
  df, mols = extract_data(args.input_file, args.input_type, args.fields,
      args.field_types, args.prediction_endpoint, args.smiles_endpoint,
      args.threshold, args.delimiter)
  print "Generating targets"
  generate_targets(df, mols, args.prediction_endpoint, args.split_endpoint,
      args.smiles_endpoint, args.id_endpoint, out_y_pkl)
  print "Generating user-specified features"
  generate_features(df, args.feature_endpoints, args.smiles_endpoint,
                    args.id_endpoint, out_x_pkl)
  print "Generating circular fingerprints"
  generate_vs_utils_features(df, args.name, args.out, args.smiles_endpoint,
      args.id_endpoint, "fingerprints")
  print "Generating rdkit descriptors"
  generate_vs_utils_features(df, args.name, args.out, args.smiles_endpoint,
      args.id_endpoint, "descriptors")
  print "Generating smiles descriptors"
  generate_smiles(df, args.name, args.out, args.smiles_endpoint, args.id_endpoint)

def train_test_input(args):
  """Saves transformed model."""
  targets = get_target_names(args.paths)
  output_transforms = {target: args.output_transforms for target in targets}
  if "smiles" in args.feature_types:
    dtype=object
  else:
    dtype=float
  train_dict, test_dict = process_datasets(args.paths,
      args.input_transforms, output_transforms, feature_types=args.feature_types, 
      splittype=args.splittype, weight_positives=args.weight_positives,
      mode=args.mode, dtype=dtype)
  trans_train_dict = transform_data(train_dict, args.input_transforms,
      args.output_transforms)
  trans_test_dict = transform_data(test_dict, args.input_transforms, args.output_transforms)
  transforms = {"input_transforms": args.input_transforms,
                "output_transform": args.output_transforms}
  stored_train = {"raw": train_dict, "transformed": trans_train_dict, "transforms": transforms}
  stored_test = {"raw": test_dict, "transformed": trans_test_dict, "transforms": transforms}
  with gzip.open(args.train_out, "wb") as f:
    pickle.dump(stored_train, f)
  with gzip.open(args.test_out, "wb") as f:
    pickle.dump(stored_test, f)

def fit_model(args):
  """Builds model from featurized data."""
  targets = get_target_names(args.paths)
  task_types = {target: args.task_type for target in targets}

  with gzip.open(args.saved_data) as f:
    stored_train = pickle.load(f)
  train_dict = stored_train["transformed"]

  if args.model == "singletask_deep_network":
    from deep_chem.models.deep import fit_singletask_mlp
    models = fit_singletask_mlp(train_dict, task_types, n_hidden=args.n_hidden,
      learning_rate=args.learning_rate, dropout=args.dropout,
      nb_epoch=args.n_epochs, decay=args.decay, batch_size=args.batch_size,
      validation_split=args.validation_split)
  elif args.model == "multitask_deep_network":
    from deep_chem.models.deep import fit_multitask_mlp
    models = fit_multitask_mlp(train_dict, task_types,
      n_hidden=args.n_hidden, learning_rate = args.learning_rate,
      dropout = args.dropout, batch_size=args.batch_size,
      nb_epoch=args.n_epochs, decay=args.decay,
      validation_split=args.validation_split)
  elif args.model == "3D_cnn":
    from deep_chem.models.deep3d import fit_3D_convolution
    models = fit_3D_convolution(train_dict, task_types,
        nb_epoch=args.n_epochs, batch_size=args.batch_size)
  elif args.model == "neural_fingerprint":
    from deep_chem.models.neural_fingerprint import fit_neural_fingerprints
    models = fit_neural_fingerprints(train_dict, task_types)
  else:
    models = fit_singletask_models(train_dict, args.model, task_types)
  if args.model in ["singletask_deep_network", "multitask_deep_network", "3D_cnn"]:
    modeltype = "keras"
  elif args.model in ["neural_fingerprint"]:
    modeltype = "autograd"
  else:
    modeltype = "sklearn"
  save_model(models, modeltype, args.saved_out)

def eval_trained_model(args):
  model = load_model(args.modeltype, args.saved_model)
  targets = get_target_names(args.paths)
  task_types = {target: args.task_type for target in targets}

  with gzip.open(args.saved_data) as f:
    stored_test = pickle.load(f)
  test_dict = stored_test["transformed"]
  raw_test_dict = stored_test["raw"]
  output_transforms = stored_test["transforms"]["output_transform"]

  results, aucs, r2s, rms = compute_model_performance(raw_test_dict, test_dict,
      task_types, model, args.modeltype, output_transforms, args.compute_aucs,
      args.compute_r2s, args.compute_rms, args.compute_recall,
      args.compute_accuracy, args.compute_matthews_corrcoef) 
  if args.csv_out is not None:
    results_to_csv(results, args.csv_out, task_type=args.task_type)

def main():
  args = parse_args()
  args.func(args)

if __name__ == "__main__":
  main()
