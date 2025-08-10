import kfp
from kfp import dsl
import kfp.kubernetes

from components import evaluate_model


@dsl.pipeline(
    name="Evaluate model",
    description="Provides simple evaluation of an LLM model and exports metrics",
)
def evaluate_model_pipeline(
    model_path: str = "ibm-granite/granite-3.3-8b-instruct",
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
        model_path: The name or path of the model to evaluate. Defaults to
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
        model_path=model_path,
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
        package_path="eval_pipeline.yaml",
    )
