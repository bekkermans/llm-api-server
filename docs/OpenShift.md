# How to deploy LLM API Server on OpenShift cluster

## Step 1: Clone the project

```bash
git clone https://github.com/bekkermans/llm-api-server
cd llm-api-server
```

## Step 2: Create an OpenShift Namespace for the project

To isolate the deployment, create a new namespace for your project. For example, to create the `app-llm-api` namespace, follow these steps:

- Create a new file called `namespace.yaml` with the following content:

```yaml
---
apiVersion: v1
kind: Namespace
metadata:
  name: app-llm-api
```

- Apply the namespace configuration using the OpenShift CLI:

```bash
oc apply -f namespace.yaml
```

## Step 3: Create persistent volume

The project requires a persistent volume to store model weights. The size of the volume depends on the number and size of the models you plan to use. For example, serving both `LLAMA2-13B` and `all-MiniLM-L6-v2` models would require at least 30GB of storage.

Follow these steps to create a 30GB persistent storage volume:

- Create a new persistent volume claim configuration file `pvc.yaml` with the following content:

```yaml
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-llm-api
  namespace: app-llm-api
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 30Gi
  storageClassName: standard-csi
```

- Apply the PVC configuration using the OpenShift CLI:

```bash
oc apply -f pvc.yaml
```

## Step 4: Prepare config file and create ConfigMap

The LLM API server is built upon the Ray framework, and its configuration file contains Ray cluster and application configuration sections. To configure LLM, you'll need to modify the `llms` section. 

Follow these steps to set up the LLM's configuration, for example, to run the API with only text encoding `sentence-transformers/all-MiniLM-L6-v2` model:

- Create a configuration file from the provided example:

```bash
cp config.yaml.examples config.yaml
```

- Modify the `config.yaml` file as follows:
  >  ❗️**Note:** If the OpenShift cluster doesn't support GPU accelerations set the `num_gpus:` as `0`

```yaml
http_options: 
  host: 0.0.0.0
  port: 7000
applications:
  - name: API
    import_path: api:app_builder
    deployments:
    - name: API
      num_replicas: 1
      ray_actor_options:
        num_gpus: 1
    args:
      llms:
        embedding:
          - name: "sentence-transformers/all-MiniLM-L6-v2"
            class: "sentence-transformers"
```

- Create the configuration file as a ConfigMap named `ray-config` using the OpenShift CLI 

```bash
oc create configmap ray-config --from-file=config.yaml -n app-llm-api
```

## Step 5: LLM API deployment

With the configuration in place and all the necessary resources set up, everything is ready to deploy the LLM API server.

Follow these steps to start the deployment:

- Create a deployment configuration file named `deployment.yaml` using the following example: 
  
```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: llm-api
  name: llm-api-server
  namespace: app-llm-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-api
  template:
    metadata:
      labels:
        app: llm-api
    spec:
      containers:
      - name: llm-api-server
        image: "quay.io/redhat_emp1/llm-api-server:latest"
        env:
          - name: HOME
            value: /home/llm-api
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 7000
          protocol: TCP
        volumeMounts:
          - mountPath: /home/llm-api
            name: llm-api-cache
          - mountPath: /llm_api_server/config.yaml
            name: llm-api-config
            subPath: config.yaml
      volumes:
        - name: llm-api-cache
          persistentVolumeClaim:
            claimName: pvc-llm-api
            readOnly: false
        - configMap:
            defaultMode: 420
            items:
            - key: config.yaml
              path: config.yaml
            name: ray-config
          name: llm-api-config
```

>  ❗️**Note:** If the OpenShift cluster doesn't support GPU accelerations remove the following section in `deployment.yaml` file:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

- Apply the Deployment configuration using the OpenShift CLI:

```bash
oc apply -f deployment.yaml
```

- Wait for the pod to enter the `Running` state. To check the pod status, use the following OpenShift CLI command:

```bash
oc get pods -n app-llm-api
```

## Step 6: Create a Service 

To make your LLM API accessible within the OpenShift cluster and from external sources, you need to create a service. Follow these steps to create a service for your deployment:

- Create a service configuration file, for example, `llm-api-svc.yaml`, with the following content:

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: llm-api-svc
  namespace: app-llm-api
  labels:
    app: llm-api
spec:
  selector:
    app: llm-api
  ports:
  - name: llm-api-server
    port: 80
    targetPort: 7000
  sessionAffinity: ClientIP
```

- Apply the service configuration using the OpenShift CLI:

```bash
oc apply -f llm-api-svc.yaml

```

- Expose the service using the OpenShift CLI:

```bash
oc expose service llm-api-svc -n app-llm-api
```

- Get the URL address using the OpenShift CLI:

```bash
oc get route llm-api-svc -n app-llm-api --template='{{ .spec.host }}'
```

This will provide the URL that you can use to access the LLM API both within and outside the OpenShift cluster.

### Congratulations! You have created a service for your LLM API, making it accessible both within and outside the OpenShift cluster.