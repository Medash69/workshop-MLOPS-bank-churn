

# # # # # # #  6.4 Script Complet : # # # # # # # # 
# RUN in cmd ubuntu bash shell : ash@MEDASH45:~$ cd "/mnt/d/Education/Workshop MLOps avec Azure/bank-churn-mlops" 
# az login
# az account list-locations --query "[?metadata.regionCategory=='Recommended'].name" -o tsv
# sed -i 's/\r$//' deploy.sh
# ./deploy.sh 

#!/usr/bin/env bash
set -euo pipefail
#################################
# VARIABLES DÉFINITIVES
#################################
RESOURCE_GROUP="rg-mlops-bank-churn"  
LOCATION="swedencentral"   # Forcé West Europe (garanti)
FALLBACK_LOCATION="northeurope"     # Fallback garanti
ACR_NAME="mlops$(whoami | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"  # 100% minuscules
CONTAINER_APP_NAME="bank-churn" 
CONTAINERAPPS_ENV="env-mlops-workshop"
IMAGE_NAME="churn-api"
IMAGE_TAG="v1"
TARGET_PORT=8000

#################################
# 0) Contexte Azure + Vérification Extensions
#################################
echo "Vérification du contexte Azure..."
az account show --query "{name:name, cloudName:cloudName}" -o json >/dev/null

echo "Vérification/installation des extensions Azure CLI..."

# Vérifier et installer containerapp si nécessaire
if ! az extension show --name containerapp >/dev/null 2>&1; then
    echo "📦 Installation de l'extension containerapp..."
    az extension add --name containerapp --upgrade -y --only-show-errors
    echo "✅ Extension containerapp installée"
else
    echo "✅ Extension containerapp déjà installée"
    # Mise à jour silencieuse
    az extension update --name containerapp -y --only-show-errors 2>/dev/null || true
fi

# Liste des extensions installées pour vérification
echo "Extensions installées :"
az extension list --query "[].{Name:name, Version:version}" -o table

#################################
# 1) Providers nécessaires
#################################
echo "Register providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.Web --wait
az provider register --namespace Microsoft.OperationalInsights --wait

#################################
# 2) Resource Group
#################################
echo "Création/validation du groupe de ressources..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null || true
echo "✅ RG OK: $RESOURCE_GROUP"

#################################
# 3) Création ACR (Simplifiée)
#################################
echo "Création du Container Registry (ACR) en $LOCATION..."

az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --location "$LOCATION"

echo "✅ ACR créé : $ACR_NAME"

#################################
# 4) Login ACR + Push image
#################################
echo "Connexion au registry..."
az acr login --name "$ACR_NAME" >/dev/null

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
echo "ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER"

# Récupération des credentials AU BON ENDROIT
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "Build + Tag + Push..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
echo "✅ Image pushée dans ACR"

#################################
# 5) Log Analytics (corrigé)
#################################
LAW_NAME="law-mlops-$(whoami)-$RANDOM"
echo "Création Log Analytics: $LAW_NAME"
az monitor log-analytics workspace create -g "$RESOURCE_GROUP" -n "$LAW_NAME" -l "$LOCATION" >/dev/null
sleep 10  # Attente nécessaire

# Commande corrigée avec paramètres explicites
LAW_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query customerId -o tsv | tr -d '\r')

LAW_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query primarySharedKey -o tsv | tr -d '\r')
echo "✅ Log Analytics OK"

#################################
# 6) Container Apps Environment
#################################
echo "Création/validation Container Apps Environment: $CONTAINERAPPS_ENV"
if ! az containerapp env show -n "$CONTAINERAPPS_ENV" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    -n "$CONTAINERAPPS_ENV" \
    -g "$RESOURCE_GROUP" \
    -l "$LOCATION" \
    --logs-workspace-id "$LAW_ID" \
    --logs-workspace-key "$LAW_KEY" >/dev/null
fi
echo "✅ Environment OK"

#################################
# 7) Déploiement Container App
#################################
echo "Déploiement Container App: $CONTAINER_APP_NAME"

# On vérifie si l'app existe
if az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  echo "L'application existe, mise à jour de l'image..."
  # Pour l'update, on ne change que l'image (les accès registre sont déjà mémorisés)
  az containerapp update \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --image "$IMAGE"
else
  echo "Création d'une nouvelle application..."
  az containerapp create \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENV" \
    --image "$IMAGE" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --min-replicas 1 \
    --max-replicas 1
fi
echo "✅ Container App OK"

#################################
# 8) URL API
#################################
APP_URL=$(az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')

echo ""
echo "=========================================="
echo "✅ DÉPLOIEMENT RÉUSSI"
echo "=========================================="
echo "ACR      : $ACR_NAME"
echo "Region   : $LOCATION"
echo "Resource Group: $RESOURCE_GROUP"
echo ""
echo "URLs de l'application :"
echo "  API      : https://$APP_URL"
echo "  Health   : https://$APP_URL/health"
echo "  Docs     : https://$APP_URL/docs"
echo ""
echo "Pour supprimer toutes les ressources :"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo "=========================================="
