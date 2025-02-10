# Use NVIDIA PyTorch container as the base image
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
# Set up the working directory
WORKDIR /workspace

# Copy requirements file and install dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

EXPOSE 8080

# Run MLflow in the background and allow overriding commands
ENTRYPOINT ["bash", "-c", "mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db & exec \"$@\""]

# Default command when no arguments are provided
CMD ["bash"]
