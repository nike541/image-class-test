import kfp
from kfp import dsl
from kfp.components import create_component_from_func
from kubernetes import client, config
import yaml


@create_component_from_func
def download_and_convert_model(model_url: str, model_name: str, model_version: int, output_path: str):
    import mlflow
    import os
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri('http://mlflow.kubeflow.svc.cluster.local:5000')
    client = MlflowClient()

    model_url = f"models:/{model_name}/{model_version}"
    local_path = os.path.join(output_path, str(model_version))
    os.makedirs(local_path, exist_ok=True)

    mlflow.pyfunc.load_model(model_url, local_path)

    config_text = f"""
    name: {model_name}
    platform: "onnxruntime"  # Change based on model type
    max_batch_size: 128
    input: [    
    {{
      "name": "input", 
      "data_type": "fp32", 
      "dims": [3, 224, 224] 
    }}
    ]
    output: [ {{
      "name": "output",
      "data_type": "fp32",
      "dims": [1000]
    }}]
    """
    with open(os.path.join(local_path, "config.pbtxt"), "w") as f:
        f.write(config_text)

    return local_path


@create_component_from_func
def deploy_triton_server(model_path: str, model_name: str, model_version: int, namespace: str, model_repository: str):
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    autoscaling_v2 = client.AutoscalingV2beta2Api()

    # Define Deployment
    deployment_yaml = f"""
    apiVersion: apps/v1
    kind: Deployment
    metadata: 
        name: tritonserver
    spec:
        replicas: 1    
        selector:
            matchLabels:
                app: tritonserver
        template:
            metadata:
                labels:
                    app: tritonserver
            spec: 
                containers:
                    - name: tritonserver
                      image: nvcr.io/nvidia/tritonserver:21.02-py3
                      resources: 
                          limits:
                              nvidia.com/gpu: 1
                              memory: 8Gi
                              cpu: 2
                      ports:
                        - containerPort: 8000
                        - containerPort: 8001
                        - containerPort: 8002
                      args:
                        - "tritonserver"
                        - "--model-repository={model_repository}"
                        - "--strict-model-config=false"
                      volumeMounts:
                        - mountPath: /models
                          name: model-repository
                volumes:
                    - name: model-repository
                      persistentVolumeClaim:
                          claimName: model-repository-pvc
    """
    deployment = yaml.safe_load(deployment_yaml)

    # Define Service
    service_yaml = f"""
    apiVersion: v1
    kind: Service
    metadata:
        name: tritonserver
    spec:
        type: LoadBalancer
        selector:
            app: tritonserver
        ports:
          - port: 8000
            targetPort: 8000
            name: http
          - port: 8001
            targetPort: 8001
            name: grpc
          - port: 8002
            targetPort: 8002
            name: metrics
    """
    service = yaml.safe_load(service_yaml)

    # Define HPA (Horizontal Pod Autoscaler)
    hpa_yaml = f"""
    apiVersion: autoscaling/v2beta2
    kind: HorizontalPodAutoscaler
    metadata:
        name: tritonserver
    spec:
        scaleTargetRef:
            apiVersion: apps/v1
            kind: Deployment
            name: tritonserver
        minReplicas: 1
        maxReplicas: 3
        metrics:
            - type: Resource
              resource:
                name: cpu
                target:
                    type: Utilization
                    averageUtilization: 80
    """
    hpa = yaml.safe_load(hpa_yaml)

    # Apply to Kubernetes Cluster
    apps_v1.create_namespaced_deployment(namespace=namespace, body=deployment)
    core_v1.create_namespaced_service(namespace=namespace, body=service)
    autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(namespace=namespace, body=hpa)


@dsl.pipeline(
    name="Deploy Triton Server",
    description="Pipeline to deploy Triton Inference Server"
)
def mlflow_triton_pipeline(
    mlflow_url: str = "",
    model_name: str = "resnet",
    model_version: int = 1,
    model_repository: str = "/mnt/models"
):
    download_and_convert_model_op = download_and_convert_model(
        model_url=mlflow_url, 
        model_name=model_name, 
        model_version=model_version, 
        output_path=model_repository
    )

    deploy_triton_server_op = deploy_triton_server(
        model_path=download_and_convert_model_op.output, 
        model_name=model_name, 
        model_version=model_version, 
        namespace="triton-server", 
        model_repository=model_repository
    )

    deploy_triton_server_op.after(download_and_convert_model_op)

    # Set resource requests and limits
    download_and_convert_model_op.set_memory_request("4G")
    download_and_convert_model_op.set_cpu_request("1")
    download_and_convert_model_op.set_memory_limit("4G")
    download_and_convert_model_op.set_cpu_limit("1")


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        mlflow_triton_pipeline,
        'mlflow_triton_pipeline.yaml'
    )
