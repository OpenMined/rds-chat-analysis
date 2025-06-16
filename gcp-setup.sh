#!/usr/bin/env bash
set -euo pipefail

# Variables — update these!
PROJECT_ID="YOUR_PROJECT_ID"
REGION="us-central1"
MOCK_DB_ROOT_PW="MOCK_PASSWORD"
PRIVATE_DB_ROOT_PW="YOUR_PASSWORD"
ANALYTICS_PW="ANALYTICS_PASSWORD"
SSH_KEY_PATH="$HOME/.ssh/notebook_vm_key"
SSH_KEY_PUB_PATH="${SSH_KEY_PATH}.pub"
SERVICE_ACCOUNT="vm-sql-sa"
FIREWALL_TAG="ssh"

# 1. Enable required APIs
gcloud config set project "$PROJECT_ID"
gcloud services enable \
  compute.googleapis.com \
  sqladmin.googleapis.com \
  iam.googleapis.com \
  servicenetworking.googleapis.com

# 2. Configure Private Services Access
gcloud compute addresses create google-managed-services-private-cloud-sql \
  --global \
  --purpose=VPC_PEERING \
  --network=default \
  --prefix-length=24

gcloud services vpc-peerings connect \
  --service=servicenetworking.googleapis.com \
  --network=default \
  --ranges=google-managed-services-private-cloud-sql \
  --project="$PROJECT_ID"

# 3. Create and permission the shared Service Account
gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
  --display-name="Notebook VM Cloud SQL SA"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudsql.instanceUser"

# 4. Create Mock Public PostgreSQL (+ pgvector)
gcloud sql instances create mock-db \
  --database-version=POSTGRES_16 \
  --region="$REGION" \
  --assign-ip \
  --authorized-networks=0.0.0.0/0 \
  --root-password="$MOCK_DB_ROOT_PW" \
  --tier=db-perf-optimized-N-2

gcloud sql connect mock-db --user=postgres --quiet \
  --command="CREATE EXTENSION IF NOT EXISTS vector;"

# 5. Create Private pgvector-Enabled PostgreSQL
gcloud sql instances create private-db \
  --database-version=POSTGRES_16 \
  --region="$REGION" \
  --no-assign-ip \
  --network=default \
  --root-password="$PRIVATE_DB_ROOT_PW" \
  --tier=db-perf-optimized-N-2 \
  --database-flags=cloudsql.iam_authentication=on

gcloud sql connect private-db --user=postgres --quiet \
  --command="CREATE EXTENSION IF NOT EXISTS vector;"

# 5.1 Add IAM-based user
gcloud sql users create \
  "${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --instance=private-db \
  --type=CLOUD_IAM_SERVICE_ACCOUNT

# 5.2 Create built-in analytics_user (Optional)
gcloud sql users create analytics_user \
  --instance=private-db \
  --password="$ANALYTICS_PW"

# 6. Firewall: allow SSH
gcloud compute firewall-rules create allow-ssh \
  --network=default \
  --action=ALLOW \
  --direction=INGRESS \
  --rules=tcp:22 \
  --target-tags="$FIREWALL_TAG"
# :contentReference[oaicite:11]{index=11}

# 7. Generate SSH key (if needed)
if [ ! -f "$SSH_KEY_PATH" ]; then
  ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "notebook_vm" 
fi

# 8. Launch Notebook VMs with SSH key, Proxy & Jupyter
for i in 1 2; do
  VM_NAME="notebook-vm-$i"
  ZONE="${REGION}-a"
  gcloud compute instances create "$VM_NAME" \
    --zone="$ZONE" \
    --machine-type=e2-medium \
    --service-account="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --scopes=cloud-platform \
    --tags="$FIREWALL_TAG" \
    --metadata="ssh-keys=youruser:$(<"$SSH_KEY_PUB_PATH")" \
    --metadata=startup-script='#!/bin/bash
      wget https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.16.0/cloud-sql-proxy.linux.amd64 \
        -O /usr/local/bin/cloud-sql-proxy && chmod +x /usr/local/bin/cloud-sql-proxy
      apt-get update && apt-get install -y python3-pip postgresql-client
      pip3 install jupyter
    '
done

echo "All resources created. To connect:"
echo "1) SSH: ssh -i $SSH_KEY_PATH youruser@$(gcloud compute instances describe notebook-vm-1 --zone=${REGION}-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo "2) Start proxy on VM: /usr/local/bin/cloud-sql-proxy --instances=${PROJECT_ID}:${REGION}:private-db --ip_address_types=PRIVATE --auto-iam-authn &"
echo "3) psql private: psql -h /cloudsql/${PROJECT_ID}:${REGION}:private-db -U \"${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com\" -d postgres"
echo "4) psql mock: psql -h \$(gcloud sql instances describe mock-db --format='get(ipAddresses[0].ipAddress)') -U postgres -d postgres"
