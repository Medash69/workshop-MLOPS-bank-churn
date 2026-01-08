#!/usr/bin/env bash
set -euo pipefail

#################################
# VARIABLES (NETTOYÉES)
#################################
RESOURCE_GROUP="rg-nlp-deployment"
LOCATION="swedencentral"
CONTAINERAPPS_ENV="env-nlp"
# On s'assure que le nom est propre
ACR_NAME="mlopsashash" 
CONTAINER_APP_NAME="bank-churn" 
IMAGE_NAME="churn-api"
IMAGE_TAG="v1"
TARGET_PORT=8000

echo "Vérification ACR..."
if ! az acr show -n "$ACR_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az acr create -g "$RESOURCE_GROUP" -n "$ACR_NAME" --sku Basic --admin-enabled true -l "$LOCATION"
fi

# NETTOYAGE DES VARIABLES ICI avec tr -d '\r'
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
echo "ACR Server: $ACR_LOGIN_SERVER"

az acr login --name "$ACR_NAME"

echo "Build et Push..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
# On s'assure que le tag est construit sans caractères cachés
FULL_IMAGE_NAME="${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME"
docker push "$FULL_IMAGE_NAME"

ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')

echo "Déploiement dans l'environnement existant..."
az containerapp create \
  -n "$CONTAINER_APP_NAME" \
  -g "$RESOURCE_GROUP" \
  --environment "$CONTAINERAPPS_ENV" \
  --image "$FULL_IMAGE_NAME" \
  --ingress external \
  --target-port "$TARGET_PORT" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --min-replicas 1 \
  --max-replicas 1

APP_URL=$(az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')
echo "=========================================="
echo "✅ TERMINÉ !"
echo "URL : https://$APP_URL"
echo "=========================================="