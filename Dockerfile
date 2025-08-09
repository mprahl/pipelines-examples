FROM registry.access.redhat.com/ubi9/python-311:latest

RUN pip install kfp==2.13.0 \
    transformers \
    torch \
    accelerate \
    datasets \
    lm-eval[vllm] \
    unitxt \
    peft \
    accelerate \
    trl \
    bitsandbytes

CMD ["python"]
