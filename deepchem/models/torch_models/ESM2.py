from typing import Dict, Any
from deepchem.models.torch_models.hf_models import HuggingFaceModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel


class ESM2(HuggingFaceModel):
    """
    ESM2 Model

    ESM-2 is a transformer model for learning on protein sequences.
    The model architecture is based on the ESM (Evolutionary Scale Modeling) architecture.
    The model can be used for finetuning ESM-2 for downstream applications.

    The model supports sequence-level tasks for regression or classification and they can be specified
    using `regression` and `classification` as arguments to the `task` keyword during model initialisation.

    The model uses a tokenizer to create input tokens for the model from the protein sequence strings.
    The default tokenizer model is loaded from huggingFace model hub (https://huggingface.co/facebook/esm2_t12_35M_UR50D).

    Parameters
    ----------
    task : str
        The task to be performed by the model. Can be either 'regression' or 'classification'.
    tokenizer_path : str, optional (default='facebook/esm2_t12_35M_UR50D')
        The path to the pretrained tokenizer model. The tokenizer path can either be a huggingFace tokenizer model or a path in the local machine containing the tokenizer.
    n_tasks : int, optional (default=1)
        The number of tasks to be performed by the model.
    config : Dict[Any, Any], optional
        A dictionary of configuration options for the model.
    **kwargs
        Additional keyword arguments to be passed to the `HuggingFaceModel` class.

    Example
    -------

    >>> import os
    >>> import tempfile
    >>> import pandas as pd
    >>> import deepchem as dc
    >>> from deepchem.feat import HuggingFaceFeaturizer
    >>> from transformers import AutoTokenizer

    >>> tempdir = tempfile.mkdtemp()

    >>> # preparing dataset
    >>> protein_sequences = ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFK",
    ...                      "GQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFK"]
    >>> labels = [1, 0]
    >>> df = pd.DataFrame(list(zip(protein_sequences, labels)), columns=["sequence", "label"])

    >>> with dc.utils.UniversalNamedTemporaryFile(mode='w') as tmpfile:
    ...     df.to_csv(tmpfile.name)
    ...     loader = dc.data.CSVLoader(["label"], feature_field="sequence", featurizer=dc.feat.DummyFeaturizer()) 
    ...     dataset = loader.create_dataset(tmpfile.name)

    >>> # finetuning in classification mode
    >>> finetune_model_dir = os.path.join(tempdir, 'finetune-model')
    >>> finetune_model = ESM2(task='classification', model_dir=finetune_model_dir)
    >>> finetuning_loss = finetune_model.fit(dataset, nb_epoch=1)

    >>> # prediction and evaluation
    >>> result = finetune_model.predict(dataset)
    >>> eval_results = finetune_model.evaluate(dataset, metrics=dc.metrics.Metric(dc.metrics.accuracy_score))
    """

    def __init__(self,
                 task: str,
                 tokenizer_path: str = 'facebook/esm2_t12_35M_UR50D',
                 n_tasks: int = 1,
                 config: Dict[Any, Any] = {},
                 **kwargs):
        self.n_tasks = n_tasks
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model: PreTrainedModel
        esm2_config = AutoConfig.from_pretrained(tokenizer_path, **config)
        esm2_config.num_labels = n_tasks

        if task == 'regression':
            esm2_config.problem_type = 'regression'
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=esm2_config)
        elif task == 'classification':
            if n_tasks == 1:
                esm2_config.problem_type = 'single_label_classification'
            else:
                esm2_config.problem_type = 'multi_label_classification'
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path, config=esm2_config)
        else:
            raise ValueError('invalid task specification')

        super(ESM2, self).__init__(model=model,
                                   task=task,
                                   tokenizer=tokenizer,
                                   **kwargs)
