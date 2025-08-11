from kfp import dsl
import kfp


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


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        evaluate_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
