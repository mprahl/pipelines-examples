import kfp
from kfp import dsl
import kfp.kubernetes


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
    model_name: str,
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
            "pretrained": model_name,
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


@dsl.pipeline(
    name="Evaluate model",
    description="Provides simple evaluation of an LLM model and exports metrics",
)
def evaluate_model_pipeline(
    model_name: str = "ibm-granite/granite-3.3-8b-instruct",
    batch_size: int = 2,
    limit: int = None,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
    add_bos_token: bool = True,
    include_classification_tasks: bool = True,
    include_summarization_tasks: bool = True,
    verbosity: str = "INFO",
    max_batch_size: int = None,
    accelerator_limit: str = "1",
    cpu_request: str = "4000m",
    memory_request: str = "100G",
):
    """Evaluates an LLM model using various tasks and exports metrics.

    This pipeline loads a specified model and evaluates it on classification and
    summarization tasks, then exports the results as metrics and artifacts.

    Args:
        model_name: The name or path of the model to evaluate. Defaults to
            "ibm-granite/granite-3.3-8b-instruct".
        batch_size: The batch size for model inference. Defaults to 2.
        limit: Maximum number of examples to evaluate per task. If None, evaluates
            all available examples. Defaults to None.
        max_model_len: Maximum sequence length for the model. Defaults to 4096.
        gpu_memory_utilization: Fraction of GPU memory to use for model loading.
            Must be between 0.0 and 1.0. Defaults to 0.8.
        dtype: Data type for model weights. Options include "bfloat16", "float16",
            "float32". Defaults to "bfloat16".
        add_bos_token: Whether to add beginning-of-sequence token to inputs.
            Defaults to True.
        include_classification_tasks: Whether to include classification tasks
            (RTE, WNLI) in evaluation. Defaults to True.
        include_summarization_tasks: Whether to include summarization tasks
            (XSum) in evaluation. Defaults to True.
        verbosity: Logging verbosity level. Options include "DEBUG", "INFO",
            "WARNING", "ERROR". Defaults to "INFO".
        max_batch_size: Maximum batch size for model inference. If None, uses
            the same value as batch_size. Defaults to None.
        accelerator_limit: Number of GPUs to request. Defaults to "1".
        cpu_request: CPU request for the pod. Defaults to "4000m".
        memory_request: Memory request for the pod. Defaults to "100G".
    """
    evaluate_model(
        model_name=model_name,
        batch_size=batch_size,
        limit=limit,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        add_bos_token=add_bos_token,
        include_classification_tasks=include_classification_tasks,
        include_summarization_tasks=include_summarization_tasks,
        verbosity=verbosity,
        max_batch_size=max_batch_size,
    ).set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(
        accelerator_limit
    ).set_cpu_request(
        cpu_request
    ).set_memory_request(
        memory_request
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=evaluate_model_pipeline,
        package_path="evaluate_model_pipeline.yaml",
    )
