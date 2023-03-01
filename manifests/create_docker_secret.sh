!bin/bash
kubectl create secret private-registry private-registry-key \
--from-file=.dockerconfigjson=/tmp/config.json \
--type=kubernetes.io/dockerconfigjson \
--namespace=ray \
-o yaml > docker-secret.yaml