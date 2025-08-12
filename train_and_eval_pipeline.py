import kfp
from kfp import dsl
import kfp.kubernetes
from typing import Optional

from components import evaluate_model, train_model


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["datasets"],
)
def prepare_dataset(
    yoda_train_dataset: dsl.Output[dsl.Dataset],
    yoda_eval_dataset: dsl.Output[dsl.Dataset],
    train_split_ratio: float = 0.8,
):
    """Prepare the training and evaluation datasets by downloading and preprocessing.

    Downloads the yoda_sentences dataset from HuggingFace, renames columns to match
    the expected format for training (prompt/completion), splits into train/eval sets,
    and saves them as output artifacts.

    Args:
        train_dataset (dsl.Output[dsl.Dataset]): Output dataset for training.
        eval_dataset (dsl.Output[dsl.Dataset]): Output dataset for evaluation.
        train_split_ratio (float): Ratio of data to use for training (0.0-1.0).
                                  Defaults to 0.8 (80% train, 20% eval).
    """
    from datasets import load_dataset

    print("Downloading and loading the dataset from dvgodoy/yoda_sentences")
    dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
    print("Renaming columns")
    dataset = dataset.rename_column("sentence", "prompt")
    dataset = dataset.rename_column("translation_extra", "completion")
    dataset = dataset.remove_columns(["translation"])

    # Add prefix to prompts
    print("Adding Yoda speak prefix to prompts")

    def add_yoda_prefix(example):
        example["prompt"] = (
            "Translate the following to Yoda speak: " + example["prompt"]
        )
        return example

    dataset = dataset.map(add_yoda_prefix)

    # Split the dataset into train and eval sets
    print(
        f"Splitting dataset with {len(dataset)} rows into train ({train_split_ratio:.1%}) and eval ({(1-train_split_ratio):.1%}) sets"
    )
    split_dataset = dataset.train_test_split(test_size=1 - train_split_ratio, seed=42)

    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train set: {len(train_dataset)} rows")
    print(f"Eval set: {len(eval_dataset)} rows")

    # Save both datasets
    print(f"Saving train dataset to {yoda_train_dataset.path}")
    train_dataset.save_to_disk(yoda_train_dataset.path)

    print(f"Saving eval dataset to {yoda_eval_dataset.path}")
    eval_dataset.save_to_disk(yoda_eval_dataset.path)


@dsl.pipeline(
    name="Train and evaluate",
    description="Provides complete training and evaluation of an LLM model",
)
def train_model_pipeline(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    train_epochs: int = 10,
    train_lora_rank: int = 8,
    train_learning_rate: float = 3e-4,
    train_batch_size: int = 16,
    train_max_length: int = 64,
    train_split_ratio: float = 0.8,
    # Training control parameters
    train_max_steps: Optional[int] = None,
    train_logging_steps: int = 10,
    train_save_steps: Optional[int] = None,
    train_save_strategy: str = "epoch",
    # Optimizer parameters
    train_optimizer: str = "adamw_torch",
    train_adam_beta1: float = 0.9,
    train_adam_beta2: float = 0.999,
    train_adam_epsilon: float = 1e-8,
    train_weight_decay: float = 0.01,
    # Performance optimization
    train_use_flash_attention: bool = False,
    # Infrastructure parameters
    train_num_nodes: int = 2,
    train_node_cpu_request: str = "2",
    train_node_gpu_request: str = "1",
    train_node_memory_request: str = "100Gi",
    trainer_runtime: str = "torch-distributed",
    # Storage parameters
    storage_class_name: Optional[str] = None,
    storage_size: str = "20Gi",
    # Evaluation parameters
    eval_batch_size: int = 2,
    eval_limit: int = None,
    eval_max_model_len: int = 4096,
    eval_gpu_memory_utilization: float = 0.8,
    eval_dtype: str = "bfloat16",
    eval_add_bos_token: bool = True,
    eval_include_classification_tasks: bool = True,
    eval_include_summarization_tasks: bool = True,
    eval_verbosity: str = "INFO",
    eval_max_batch_size: int = None,
    eval_num_gpus: str = "1",
    eval_cpu_request: str = "4000m",
    eval_memory_request: str = "100G",
):
    """Complete pipeline for fine-tuning and evaluating a large language model.

    This pipeline orchestrates the complete workflow for fine-tuning a large language
    model using distributed training with LoRA fine-tuning, followed by comprehensive
    evaluation. It handles dataset preparation, storage provisioning, model training,
    evaluation, and cleanup.

    Args:
        model_name (str, optional): HuggingFace model to fine-tune.
            Defaults to "meta-llama/Llama-3.2-3B-Instruct".
        train_epochs (int, optional): Number of training epochs. Defaults to 10.
        train_lora_rank (int, optional): LoRA adapter complexity (8=efficient, 16=more expressive).
            Defaults to 8.
        train_learning_rate (float, optional): Training learning rate (3e-4 is a good starting point).
            Defaults to 3e-4.
        train_batch_size (int, optional): Batch size per GPU (16 works well for most setups).
            Defaults to 16.
        train_max_length (int, optional): Maximum input sequence length (64=short, 128=medium, 256=long).
            Defaults to 64.
        train_split_ratio (float, optional): Ratio of dataset to use for training (0.0-1.0).
            Defaults to 0.8 (80% train, 20% eval).
        train_max_steps (int, optional): Maximum training steps. If specified, overrides epochs. Defaults to None.
        train_logging_steps (int, optional): Steps between logging outputs. Defaults to 10.
        train_save_steps (int, optional): Steps between model checkpoints. Defaults to None.
        train_save_strategy (str, optional): Checkpoint strategy ("epoch" or "steps"). Defaults to "epoch".
        train_optimizer (str, optional): Optimizer ("adamw_torch", "adamw_torch_fused"). Defaults to "adamw_torch".
        train_adam_beta1 (float, optional): Adam beta1 parameter. Defaults to 0.9.
        train_adam_beta2 (float, optional): Adam beta2 parameter. Defaults to 0.999.
        train_adam_epsilon (float, optional): Adam epsilon parameter. Defaults to 1e-8.
        train_weight_decay (float, optional): Weight decay for regularization. Defaults to 0.01.
        train_use_flash_attention (bool, optional): Enable Flash Attention 2. Defaults to False.
        train_num_nodes (int, optional): Number of training nodes (2=basic distributed, 4+=large scale).
            Defaults to 2.
        train_node_cpu_request (str, optional): CPU request per node (e.g., "2", "4"). Defaults to "2".
        train_node_gpu_request (str, optional): GPU request per node (e.g., "1", "2"). Defaults to "1".
        train_node_memory_request (str, optional): Memory per node (100Gi=basic, 200Gi+=memory-intensive models).
            Defaults to "100Gi".
        trainer_runtime (str, optional): Runtime to use for Kubeflow Trainer. Defaults to "torch-distributed".
        storage_class_name (str, optional): Storage class name for PVC creation. Defaults to None.
        storage_size (str, optional): Storage size for PVC creation. Defaults to "20Gi".
        eval_batch_size (int, optional): Batch size for model evaluation. Defaults to 2.
        eval_limit (int, optional): Maximum number of examples to evaluate per task. If None, evaluates all available examples. Defaults to None.
        eval_max_model_len (int, optional): Maximum sequence length for the model during evaluation. Defaults to 4096.
        eval_gpu_memory_utilization (float, optional): Fraction of GPU memory to use for model loading during evaluation. Must be between 0.0 and 1.0. Defaults to 0.8.
        eval_dtype (str, optional): Data type for model weights during evaluation. Options include "bfloat16", "float16", "float32". Defaults to "bfloat16".
        eval_add_bos_token (bool, optional): Whether to add beginning-of-sequence token to inputs during evaluation. Defaults to True.
        eval_include_classification_tasks (bool, optional): Whether to include classification tasks (RTE, WNLI) in evaluation. Defaults to True.
        eval_include_summarization_tasks (bool, optional): Whether to include summarization tasks (XSum) in evaluation. Defaults to True.
        eval_verbosity (str, optional): Logging verbosity level during evaluation. Options include "DEBUG", "INFO", "WARNING", "ERROR". Defaults to "INFO".
        eval_max_batch_size (int, optional): Maximum batch size for model inference during evaluation. If None, uses the same value as eval_batch_size. Defaults to None.
        eval_num_gpus (str, optional): Number of GPUs to request for evaluation. Defaults to "1".
        eval_cpu_request (str, optional): CPU request for the evaluation pod. Defaults to "4000m".
        eval_memory_request (str, optional): Memory request for the evaluation pod. Defaults to "100G".
    Returns:
        Pipeline object that can be compiled and executed by Kubeflow Pipelines.

    Example:
        >>> pipeline = train_model_pipeline(
        ...     model_name="meta-llama/Llama-3.2-3B-Instruct",
        ...     train_epochs=5,
        ...     train_lora_rank=16
        ... )
        >>> kfp.compiler.Compiler().compile(pipeline, "my_pipeline.yaml")
    """
    prepare_dataset_op = (
        prepare_dataset(
            train_split_ratio=train_split_ratio,
        )
        .set_caching_options(enable_caching=False)
        .set_retry(3)
    )

    create_pvc_op = kfp.kubernetes.CreatePVC(
        access_modes=["ReadWriteMany"],
        size=storage_size,
        storage_class_name=storage_class_name,
    )

    train_model_op = (
        train_model(
            pvc_name=create_pvc_op.output,
            model_name=model_name,
            epochs=train_epochs,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            input_dataset=prepare_dataset_op.outputs["yoda_train_dataset"],
            pvc_path="/workspace",
            # Pass through configuration parameters
            lora_rank=train_lora_rank,
            learning_rate=train_learning_rate,
            batch_size=train_batch_size,
            max_length=train_max_length,
            # Training control parameters
            max_steps=train_max_steps,
            logging_steps=train_logging_steps,
            save_steps=train_save_steps,
            save_strategy=train_save_strategy,
            # Optimizer parameters
            optimizer=train_optimizer,
            adam_beta1=train_adam_beta1,
            adam_beta2=train_adam_beta2,
            adam_epsilon=train_adam_epsilon,
            weight_decay=train_weight_decay,
            # Performance optimization
            use_flash_attention=train_use_flash_attention,
            # Infrastructure parameters
            num_nodes=train_num_nodes,
            train_node_cpu_request=train_node_cpu_request,
            train_node_gpu_request=train_node_gpu_request,
            train_node_memory_request=train_node_memory_request,
            trainer_runtime=trainer_runtime,
            save_merged_model_path="/workspace/merged_model",
            hf_token_secret_name="hf-token",
        )
        .after(prepare_dataset_op)
        .set_caching_options(enable_caching=False)
    )

    kfp.kubernetes.mount_pvc(
        task=train_model_op,
        pvc_name=create_pvc_op.output,
        mount_path="/workspace",
    )

    eval_model_op = (
        evaluate_model(
            model_path="/workspace/merged_model",
            batch_size=eval_batch_size,
            limit=eval_limit,
            max_model_len=eval_max_model_len,
            gpu_memory_utilization=eval_gpu_memory_utilization,
            dtype=eval_dtype,
            add_bos_token=eval_add_bos_token,
            include_classification_tasks=eval_include_classification_tasks,
            include_summarization_tasks=eval_include_summarization_tasks,
            verbosity=eval_verbosity,
            max_batch_size=eval_max_batch_size,
            custom_translation_dataset=prepare_dataset_op.outputs["yoda_eval_dataset"],
        )
        .set_caching_options(enable_caching=False)
        .set_accelerator_type("nvidia.com/gpu")
        .set_accelerator_limit(eval_num_gpus)
        .set_cpu_request(eval_cpu_request)
        .set_memory_request(eval_memory_request)
    ).after(train_model_op)
    kfp.kubernetes.mount_pvc(
        task=eval_model_op,
        pvc_name=create_pvc_op.output,
        mount_path="/workspace",
    )

    kfp.kubernetes.DeletePVC(pvc_name=create_pvc_op.output).after(eval_model_op)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=train_model_pipeline,
        package_path="train_and_eval_pipeline.yaml",
    )
