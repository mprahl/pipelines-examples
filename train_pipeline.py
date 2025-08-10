import kfp
from kfp import dsl
import kfp.kubernetes
from typing import Optional


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["kubernetes"],
)
def train_model(
    model_name: str,
    pvc_name: str,
    run_id: str,
    dataset_path: str,
    pvc_path: str,
    output_model: dsl.Output[dsl.Model],
    output_metrics: dsl.Output[dsl.Metrics],
    # Training configuration parameters
    epochs: int = 10,
    lora_rank: int = 8,
    learning_rate: float = 3e-4,
    batch_size: int = 16,
    max_length: int = 64,
    # Training control parameters
    max_steps: Optional[int] = None,
    logging_steps: int = 10,
    save_steps: Optional[int] = None,
    save_strategy: str = "epoch",
    # Optimizer parameters
    optimizer: str = "adamw_torch",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    weight_decay: float = 0.01,
    # Performance optimization
    use_flash_attention: bool = False,
    # Infrastructure parameters
    num_nodes: int = 2,
    train_node_cpu_request: str = "2",
    train_node_gpu_request: str = "1",
    train_node_memory_request: str = "100Gi",
    trainer_runtime: str = "torch-distributed",
    save_merged_model_path: str = None,
):
    """Train a large language model using distributed training with LoRA fine-tuning.

    This function creates and manages a Kubernetes TrainJob for distributed training
    of a large language model using LoRA (Low-Rank Adaptation) fine-tuning. It handles
    the complete training workflow including job creation, monitoring, and artifact
    collection.

    Args:
        model_name (str): HuggingFace model identifier (e.g., "ibm-granite/granite-3.3-8b-instruct").
        pvc_name (str): Name of the Persistent Volume Claim for data storage that is mounted to this component.
        run_id (str): Unique identifier for this training run. Use dsl.PIPELINE_JOB_ID_PLACEHOLDER.
        dataset_path (str): Path to the training dataset within the PVC.
        pvc_path (str): Base path within the PVC for storing outputs.
        output_model (dsl.Output[dsl.Model]): Kubeflow output artifact for the trained model.
        output_metrics (dsl.Output[dsl.Metrics]): Kubeflow output artifact for training metrics.
        epochs (int, optional): Number of training epochs. Defaults to 10.
        lora_rank (int, optional): LoRA adapter rank (lower = fewer parameters, faster training). Defaults to 8.
        learning_rate (float, optional): Learning rate for training optimization. Defaults to 3e-4.
        batch_size (int, optional): Per-device training batch size. Defaults to 16.
        max_length (int, optional): Maximum token sequence length for training. Defaults to 64.
        max_steps (int, optional): Maximum number of training steps. If specified, overrides epochs. Defaults to None.
        logging_steps (int, optional): Number of steps between logging outputs. Defaults to 10.
        save_steps (int, optional): Number of steps between model checkpoints. Defaults to None.
        save_strategy (str, optional): Checkpoint saving strategy ("epoch" or "steps"). Defaults to "epoch".
        optimizer (str, optional): Optimizer to use (e.g., "adamw_torch", "adamw_torch_fused"). Defaults to "adamw_torch".
        adam_beta1 (float, optional): Beta1 parameter for Adam optimizer. Defaults to 0.9.
        adam_beta2 (float, optional): Beta2 parameter for Adam optimizer. Defaults to 0.999.
        adam_epsilon (float, optional): Epsilon parameter for Adam optimizer. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 0.01.
        use_flash_attention (bool, optional): Whether to use Flash Attention 2 for improved performance. Defaults to False.
        num_nodes (int, optional): Number of nodes for distributed training. Defaults to 2.
        train_node_cpu_request (str, optional): CPU request per node (e.g., "2", "4"). Defaults to "2".
        train_node_gpu_request (str, optional): GPU request per node (e.g., "1", "2"). Defaults to "1".
        train_node_memory_request (str, optional): Memory request per node (e.g., "100Gi", "200Gi"). Defaults to "100Gi".
        trainer_runtime (str, optional): Runtime to use for Kubeflow Trainer. Defaults to "torch-distributed".
        save_merged_model_path (str, optional): Path to save the merged model (base + LoRA adapter). Useful for saving to the PVC for evaluation. Defaults to None.
    """
    import json
    import os
    import shutil
    import textwrap
    import time
    import inspect

    from kubernetes import client as k8s_client, config
    from kubernetes.client.rest import ApiException

    def get_target_modules(model_name: str) -> list:
        """Get appropriate LoRA target modules based on model architecture.

        Selects optimal layers for LoRA adaptation based on research findings:
        - Attention layers (q_proj, k_proj, v_proj, o_proj) control attention patterns
        - MLP layers (gate_proj, up_proj, down_proj) store task-specific knowledge

        Model-specific targeting:
        - Granite: Attention layers only (q,k,v,o)
        - LLaMA/Mistral/Qwen: Full coverage (attention + MLP)
        - Phi: Uses 'dense' instead of 'o_proj'
        - Unknown: Conservative fallback (q,v)

        Based on LoRA (arXiv:2106.09685), QLoRA (arXiv:2305.14314), and model-specific research.
        """
        model_name_lower = model_name.lower()

        if "granite" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        elif "llama" in model_name_lower:
            return [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif "qwen" in model_name_lower:
            return [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif "phi" in model_name_lower:
            return ["q_proj", "v_proj", "k_proj", "dense"]
        else:
            print(
                f"Warning: Unknown model architecture for {model_name}, using conservative LoRA targets"
            )
            return ["q_proj", "v_proj"]

    def train_model_func(
        lora_rank: int,
        learning_rate: float,
        batch_size: int,
        max_length: int,
        model_name: str,
        dataset_path: str,
        epochs: int,
        pvc_path: str,
        target_modules: list,
        max_steps: int,
        logging_steps: int,
        save_steps: int,
        save_strategy: str,
        optimizer: str,
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
        weight_decay: float,
        use_flash_attention: bool,
        save_merged_model_path: str = None,
    ):
        import os
        import json
        import torch
        from datasets import load_from_disk
        from peft import get_peft_model, LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        print(
            f"Worker info - Local rank: {local_rank}, World rank: {world_rank}, World size: {world_size}"
        )

        is_main_worker = world_rank == 0

        class MetricsCallback(TrainerCallback):
            def __init__(self, is_main_worker):
                self.is_main_worker = is_main_worker
                self.initial_loss = None
                self.final_loss = None

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and self.is_main_worker and "loss" in logs:
                    if self.initial_loss is None:
                        self.initial_loss = logs["loss"]
                    self.final_loss = logs["loss"]

        metrics_callback = MetricsCallback(is_main_worker)

        print("Downloading and loading model")
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        print(f"Using LoRA target modules for {model_name}: {target_modules}")

        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)

        print("Loading dataset")
        dataset = load_from_disk(dataset_path)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        sft_config = SFTConfig(
            ## Memory optimization
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=1,
            per_device_train_batch_size=batch_size,
            auto_find_batch_size=True,
            ## Dataset configuration
            max_length=max_length,
            packing=use_flash_attention,  # Packing works best with Flash Attention
            ## Training parameters
            num_train_epochs=epochs if max_steps is None else None,
            max_steps=-1 if max_steps is None else max_steps,
            learning_rate=learning_rate,
            optim=optimizer,
            ## Optimizer parameters
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            weight_decay=weight_decay,
            ## Logging and saving
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_strategy=save_strategy,
            logging_dir="./logs",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=dataset,
            callbacks=[metrics_callback],
        )

        train_result = trainer.train()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            print(f"Worker {world_rank} - Training completed and synchronized")

        if not is_main_worker:
            print(
                f"Worker {world_rank} - Skipping model export and metrics (not main worker)"
            )
            # Clean up distributed process group for non-main workers
            if torch.distributed.is_initialized():
                print(f"Worker {world_rank} - Cleaning up distributed process group")
                torch.distributed.destroy_process_group()
                print(f"Worker {world_rank} - Distributed process group destroyed")
            return

        print("Main worker (rank 0) - Exporting model and metrics...")

        # Save LoRA adapter
        model_output_path = os.path.join(pvc_path, "adapter")
        model.save_pretrained(model_output_path)
        tokenizer.save_pretrained(model_output_path)
        print("LoRA adapter exported successfully!")

        # Merge LoRA adapter with base model and save merged model for evaluation
        if save_merged_model_path:
            print(
                f"Merging LoRA adapter with base model and saving to {save_merged_model_path}..."
            )
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(save_merged_model_path)
            tokenizer.save_pretrained(save_merged_model_path)
            print(f"Merged model saved to {save_merged_model_path}")

        # Clean up distributed process group for main worker AFTER model saving
        if torch.distributed.is_initialized():
            print(f"Worker {world_rank} - Cleaning up distributed process group")
            torch.distributed.destroy_process_group()
            print(f"Worker {world_rank} - Distributed process group destroyed")

        print(f"Collecting essential metrics")
        metrics_dict = {}

        if hasattr(train_result, "train_loss"):
            metrics_dict["final_train_loss"] = train_result.train_loss
        if hasattr(train_result, "train_runtime"):
            metrics_dict["train_runtime_seconds"] = train_result.train_runtime
        if hasattr(train_result, "train_samples_per_second"):
            metrics_dict["throughput_samples_per_sec"] = (
                train_result.train_samples_per_second
            )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics_dict["total_parameters_millions"] = total_params / 1_000_000
        metrics_dict["trainable_parameters_millions"] = trainable_params / 1_000_000
        metrics_dict["lora_efficiency_percent"] = (
            trainable_params / total_params
        ) * 100

        metrics_dict["lora_rank"] = config.r
        metrics_dict["learning_rate"] = sft_config.learning_rate
        metrics_dict["effective_batch_size"] = (
            sft_config.per_device_train_batch_size * world_size
        )
        metrics_dict["dataset_size"] = len(dataset)

        metrics_dict["num_nodes"] = (
            world_size // torch.cuda.device_count()
            if torch.cuda.is_available() and torch.cuda.device_count() > 0
            else 1
        )
        if torch.cuda.is_available():
            metrics_dict["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / (
                1024**3
            )

        if metrics_callback.initial_loss and metrics_callback.final_loss:
            metrics_dict["initial_loss"] = metrics_callback.initial_loss
            metrics_dict["loss_reduction"] = (
                metrics_callback.initial_loss - metrics_callback.final_loss
            )
            metrics_dict["loss_reduction_percent"] = (
                (metrics_callback.initial_loss - metrics_callback.final_loss)
                / metrics_callback.initial_loss
            ) * 100

        with open(os.path.join(pvc_path, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=2)

        print(
            f"Exported {len(metrics_dict)} metrics to {os.path.join(pvc_path, 'metrics.json')}"
        )
        print("Model and metrics exported successfully!")

    print("=== Starting TrainJob creation process ===")

    target_modules = get_target_modules(model_name)
    print(f"Selected LoRA target modules for {model_name}: {target_modules}")

    with open(
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r"
    ) as ns_file:
        namespace = ns_file.readline()

    print("Generating command...")

    func_code = inspect.getsource(train_model_func)
    func_code = textwrap.dedent(func_code)

    func_call_code = f"""
import os
import json

# Parse function arguments from environment variable
config_json = os.environ.get("TRAINING_CONFIG", "{{}}")
func_args = json.loads(config_json)

# Call the training function with parsed arguments
{train_model_func.__name__}(**func_args)
"""

    func_code = f"{func_code}\n{func_call_code}"

    # Build package list based on configuration
    packages = ["transformers", "peft", "accelerate", "trl"]
    if use_flash_attention:
        packages.append("flash-attn")
    packages_str = " ".join(packages)

    install_script = f"""set -e
set -o pipefail

echo "=== Starting container setup ==="
echo "Python version: $(python --version)"

if ! [ -x "$(command -v pip)" ]; then
    echo "Installing pip..."
    python -m ensurepip || python -m ensurepip --user
fi

echo "Installing Python packages..."
PIP_DISABLE_PIP_VERSION_CHECK=1 python -m pip install --user --quiet --no-warn-script-location {packages_str}

echo "Creating training script..."
cat > ephemeral_component.py << 'EOF'
{func_code}
EOF

echo "Starting distributed training..."
torchrun ephemeral_component.py"""

    command = ["bash", "-c", install_script]

    print(f"Generated command: {command}")
    print(f"Command length: {len(command)}")
    print(f"Command type: {type(command)}")

    print("Loading Kubernetes configuration...")
    try:
        config.load_incluster_config()
        print("Loaded in-cluster Kubernetes configuration")
    except config.ConfigException:
        config.load_kube_config()
        print("Loaded kubeconfig Kubernetes configuration")

    print("Creating Kubernetes API client...")
    api_client = k8s_client.ApiClient()
    custom_objects_api = k8s_client.CustomObjectsApi(api_client)
    print("Successfully created Kubernetes API client")

    print("Defining TrainJob resource...")
    train_job = {
        "apiVersion": "trainer.kubeflow.org/v1alpha1",
        "kind": "TrainJob",
        "metadata": {"name": f"kfp-{run_id}", "namespace": namespace},
        "spec": {
            "runtimeRef": {"name": trainer_runtime},
            "trainer": {
                "numNodes": num_nodes,
                "resourcesPerNode": {
                    "requests": {
                        "cpu": train_node_cpu_request,
                        "memory": train_node_memory_request,
                        "nvidia.com/gpu": train_node_gpu_request,
                    },
                    "limits": {
                        "cpu": train_node_cpu_request,
                        "memory": train_node_memory_request,
                        "nvidia.com/gpu": train_node_gpu_request,
                    },
                },
                "env": [
                    {"name": "HOME", "value": "/tmp"},
                    {
                        "name": "TRAINING_CONFIG",
                        "value": json.dumps(
                            {
                                "lora_rank": lora_rank,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "max_length": max_length,
                                "model_name": model_name,
                                "dataset_path": dataset_path,
                                "epochs": epochs,
                                "pvc_path": pvc_path,
                                "target_modules": target_modules,
                                "max_steps": max_steps,
                                "logging_steps": logging_steps,
                                "save_steps": save_steps,
                                "save_strategy": save_strategy,
                                "optimizer": optimizer,
                                "adam_beta1": adam_beta1,
                                "adam_beta2": adam_beta2,
                                "adam_epsilon": adam_epsilon,
                                "weight_decay": weight_decay,
                                "use_flash_attention": use_flash_attention,
                                "save_merged_model_path": save_merged_model_path,
                            }
                        ),
                    },
                ],
                "command": command,
            },
            "podSpecOverrides": [
                {
                    "targetJobs": [{"name": "node"}],
                    "volumes": [
                        {
                            "name": "dataset-pvc",
                            "persistentVolumeClaim": {"claimName": pvc_name},
                        }
                    ],
                    "containers": [
                        {
                            "name": "node",
                            "volumeMounts": [
                                {"name": "dataset-pvc", "mountPath": "/workspace"}
                            ],
                        }
                    ],
                }
            ],
        },
    }

    print(f"TrainJob definition created:")
    print(f"  - Name: kfp-{run_id}")
    print(f"  - Namespace: {namespace}")

    print(f"  - Runtime: {trainer_runtime}")
    print(f"  - Nodes: {num_nodes}")
    print(f"  - PVC: {pvc_name}")
    print(f"  - Model: {model_name}")
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Epochs: {epochs}")

    print("Submitting TrainJob to Kubernetes...")
    try:
        response = custom_objects_api.create_namespaced_custom_object(
            group="trainer.kubeflow.org",
            version="v1alpha1",
            namespace=namespace,
            plural="trainjobs",
            body=train_job,
        )
        job_name = response["metadata"]["name"]
        print(f"TrainJob {job_name} created successfully")
        print(f"Response metadata: {response.get('metadata', {})}")
    except ApiException as e:
        print(f"Error creating TrainJob: {e}")
        print(f"Error details: {e.body}")
        print(f"Error status: {e.status}")
        raise

    print(f"Starting to monitor TrainJob {job_name} status...")
    check_count = 0
    while True:
        check_count += 1
        try:
            print(f"Checking job status (attempt {check_count})...")
            job_status = custom_objects_api.get_namespaced_custom_object(
                group="trainer.kubeflow.org",
                version="v1alpha1",
                namespace=namespace,
                plural="trainjobs",
                name=job_name,
            )

            status = job_status.get("status", {})
            conditions = status.get("conditions", [])
            print(f"Job status conditions: {conditions}")

            completed = False
            failed = False

            for condition in conditions:
                condition_type = condition.get("type", "")
                condition_status = condition.get("status", "")
                condition_reason = condition.get("reason", "")
                condition_message = condition.get("message", "")

                print(
                    f"Condition: type={condition_type}, status={condition_status}, reason={condition_reason}"
                )

                if condition_type == "Complete" and condition_status == "True":
                    print(
                        f"Training job {job_name} completed successfully: {condition_message}"
                    )
                    completed = True
                    break
                elif condition_type == "Failed" and condition_status == "True":
                    print(f"Training job {job_name} failed: {condition_message}")
                    failed = True
                    break
                elif condition_type == "Cancelled" and condition_status == "True":
                    print(f"Training job {job_name} was cancelled: {condition_message}")
                    failed = True
                    break

            if completed:
                break
            elif failed:
                raise RuntimeError(f"Training job {job_name} failed or was cancelled")
            else:
                print(f"Job is still running, continuing to wait...")

        except ApiException as e:
            print(f"Error checking job status: {e}")
            print(f"Error details: {e.body}")

        print(f"Waiting 10 seconds before next check...")
        time.sleep(10)

    print(f"Training job {job_name} completed. Logs would be retrieved here.")

    print("Processing training results...")

    metrics_file_path = os.path.join(pvc_path, "metrics.json")
    print(f"Looking for metrics file at: {metrics_file_path}")
    if os.path.exists(metrics_file_path):
        print(f"Found metrics file, reading from {metrics_file_path}")
        with open(metrics_file_path, "r") as f:
            metrics_dict = json.load(f)

        print(f"Loaded {len(metrics_dict)} metrics from file")

        exported_count = 0
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, (int, float)):
                output_metrics.log_metric(metric_name, metric_value)
                print(f"Exported metric: {metric_name} = {metric_value}")
                exported_count += 1

        print(f"Successfully exported {exported_count} metrics to Kubeflow")
        os.remove(metrics_file_path)
    else:
        print(f"Warning: Metrics file {metrics_file_path} not found")

    print("Copying model from PVC to Kubeflow output path...")
    model_source = os.path.join(pvc_path, "adapter")
    print(f"Model source: {model_source}")
    print(f"Destination: {output_model.path}")

    if not os.path.exists(model_source):
        raise FileNotFoundError(
            f"Trained model not found at expected location: {model_source}"
        )

    required_files = ["adapter_config.json", "adapter_model.bin"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_source, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    if missing_files:
        print(f"Warning: Missing expected model files: {missing_files}")
        print(f"Available files in {model_source}:")
        try:
            for file in os.listdir(model_source):
                print(f"  - {file}")
        except Exception as e:
            print(f"  Error listing files: {e}")

    output_model.name = f"{model_name}-adapter"
    shutil.copytree(model_source, output_model.path, dirs_exist_ok=True)
    print(f"Model copied successfully from {model_source} to {output_model.path}")

    print("=== TrainJob process completed successfully ===")


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=[
        "transformers",
        "torch",
        "accelerate",
        "lm-eval[vllm]",
        "unitxt",
    ],
)
def evaluate_model(
    model_path: str,
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    batch_size: int = 1,
    limit: int = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
    add_bos_token: bool = True,
    include_classification_tasks: bool = True,
    include_summarization_tasks: bool = True,
    verbosity: str = "INFO",
    max_batch_size: int = None,
):
    import logging
    import os
    import json
    import time
    import shutil
    from typing import List, Dict, Any

    from lm_eval.tasks.unitxt import task
    from lm_eval.api.registry import get_model
    from lm_eval.api.model import LM
    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import get_task_dict
    import torch

    TASK_CONFIGS = {
        "classification": [
            {
                "task": "classification_rte_simple",
                "recipe": "card=cards.rte,template=templates.classification.multi_class.relation.simple",
                "group": "classification",
                "output_type": "generate_until",
            },
            {
                "task": "classification_rte_default",
                "recipe": "card=cards.rte,template=templates.classification.multi_class.relation.default",
                "group": "classification",
                "output_type": "generate_until",
            },
            {
                "task": "classification_rte_wnli",
                "recipe": "card=cards.wnli,template=templates.classification.multi_class.relation.simple",
                "group": "classification",
                "output_type": "generate_until",
            },
        ],
        "summarization": [
            {
                "task": "summarization_xsum_formal",
                "recipe": "card=cards.xsum,template=templates.summarization.abstractive.formal,num_demos=0",
                "group": "summarization",
                "output_type": "generate_until",
            }
        ],
    }

    logging.basicConfig(
        level=getattr(logging, verbosity.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    logger.info("Validating parameters...")

    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")

    if not (0.0 <= gpu_memory_utilization <= 1.0):
        raise ValueError("gpu_memory_utilization must be between 0.0 and 1.0")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive or None")

    if not include_classification_tasks and not include_summarization_tasks:
        raise ValueError(
            "At least one of include_classification_tasks or include_summarization_tasks must be True"
        )

    logger.info("Parameter validation passed")

    logger.info("Creating tasks...")
    start_time = time.time()

    eval_tasks = []

    if include_classification_tasks:
        logger.info("Adding classification tasks...")
        classification_configs = TASK_CONFIGS["classification"]

        for config in classification_configs:
            task_obj = task.Unitxt(config=config)
            # TODO: Remove after https://github.com/EleutherAI/lm-evaluation-harness/pull/3225 is merged.
            task_obj.config.task = config["task"]
            eval_tasks.append(task_obj)

    if include_summarization_tasks:
        logger.info("Adding summarization tasks...")
        summarization_config = TASK_CONFIGS["summarization"][0]

        task_obj = task.Unitxt(config=summarization_config)
        # TODO: Remove after https://github.com/EleutherAI/lm-evaluation-harness/pull/3225 is merged.
        task_obj.config.task = summarization_config["task"]
        eval_tasks.append(task_obj)

    task_dict = get_task_dict(eval_tasks)
    logger.info(f"Created {len(eval_tasks)} tasks in {time.time() - start_time:.2f}s")

    logger.info("Loading model...")
    start_time = time.time()

    try:
        model_args = {
            "add_bos_token": add_bos_token,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "pretrained": model_path,
        }

        model_class = get_model("vllm")
        additional_config = {
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
            "device": None,
        }

        loaded_model = model_class.create_from_arg_obj(model_args, additional_config)
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    logger.info("Starting evaluation...")
    start_time = time.time()

    results = evaluate(
        lm=loaded_model,
        task_dict=task_dict,
        limit=limit,
        verbosity=verbosity,
    )

    logger.info(f"Evaluation completed in {time.time() - start_time:.2f}s")

    logger.info("Saving results...")

    def clean_for_json(obj):
        """Recursively clean objects to make them JSON serializable."""
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert non-serializable objects to string representation
            return str(obj)

    clean_results = clean_for_json(results)

    output_results.name = "results.json"

    with open(output_results.path, "w") as f:
        json.dump(clean_results, f, indent=2)
    logger.info(f"Results saved to {output_results.path}")

    logger.info("Logging metrics...")

    for task_name, task_results in clean_results["results"].items():
        for metric_name, metric_value in task_results.items():
            if isinstance(metric_value, (int, float)):
                # Skip metrics that are 0 due to a bug in the RHOAI UI.
                # TODO: Fix RHOAI UI to handle 0 values.
                # TODO: Ignore store_session_info from metrics in RHOAI UI.
                if metric_value == 0:
                    continue

                metric_key = f"{task_name}_{metric_name}"
                output_metrics.log_metric(metric_key, metric_value)
                logger.debug(f"Logged metric: {metric_key} = {metric_value}")

    logger.info("Metrics logged successfully")

    logger.info("Pipeline completed successfully")


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["datasets"],
)
def prepare_dataset(output_path: str):
    """Prepare the training dataset by downloading and preprocessing.

    Downloads the yoda_sentences dataset from HuggingFace, renames columns to match
    the expected format for training (prompt/completion), and saves it to disk.

    Args:
        output_path (str): Directory path where the processed dataset will be saved.
    """
    import os
    from datasets import load_dataset

    os.makedirs(output_path, exist_ok=True)

    print("Downloading and loading the dataset from dvgodoy/yoda_sentences")
    dataset = load_dataset("dvgodoy/yoda_sentences", split="train")
    print("Renaming columns")
    dataset = dataset.rename_column("sentence", "prompt")
    dataset = dataset.rename_column("translation_extra", "completion")
    dataset = dataset.remove_columns(["translation"])

    print("Saving dataset to", output_path)
    dataset.save_to_disk(output_path)


@dsl.pipeline(
    name="Train and evaluate",
    description="Provides complete training and evaluation of an LLM model",
)
def train_model_pipeline(
    model_name: str = "ibm-granite/granite-3.3-8b-instruct",
    train_epochs: int = 10,
    train_lora_rank: int = 8,
    train_learning_rate: float = 3e-4,
    train_batch_size: int = 16,
    train_max_length: int = 64,
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
            Defaults to "ibm-granite/granite-3.3-8b-instruct".
        train_epochs (int, optional): Number of training epochs. Defaults to 10.
        train_lora_rank (int, optional): LoRA adapter complexity (8=efficient, 16=more expressive).
            Defaults to 8.
        train_learning_rate (float, optional): Training learning rate (3e-4 is a good starting point).
            Defaults to 3e-4.
        train_batch_size (int, optional): Batch size per GPU (16 works well for most setups).
            Defaults to 16.
        train_max_length (int, optional): Maximum input sequence length (64=short, 128=medium, 256=long).
            Defaults to 64.
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
        ...     model_name="ibm-granite/granite-3.3-8b-instruct",
        ...     train_epochs=5,
        ...     train_lora_rank=16
        ... )
        >>> kfp.compiler.Compiler().compile(pipeline, "my_pipeline.yaml")
    """
    create_pvc_op = kfp.kubernetes.CreatePVC(
        access_modes=["ReadWriteMany"],
        size=storage_size,
        storage_class_name=storage_class_name,
    )

    prepare_dataset_op = (
        prepare_dataset(output_path="/workspace/dataset")
        .set_caching_options(enable_caching=False)
        .set_retry(3)
    )
    kfp.kubernetes.mount_pvc(
        task=prepare_dataset_op,
        pvc_name=create_pvc_op.output,
        mount_path="/workspace",
    )

    train_model_op = (
        train_model(
            pvc_name=create_pvc_op.output,
            model_name=model_name,
            epochs=train_epochs,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            dataset_path="/workspace/dataset",
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
        package_path="train_model_pipeline.yaml",
    )
