#!/bin/sh
# Copy local data/, artifacts/, models/ into backend PVC via kubectl cp.
# Usage: ./scripts/k8s_seed_data.sh <namespace> <release-name>
set -e
NS="${1:-saint-dev}"
RELEASE="${2:-saint-dev-saint-ops}"
POD=$(kubectl get pod -n "$NS" -l "app.kubernetes.io/component=backend" -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD" ]; then
  echo "No backend pod in namespace $NS"
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Seeding into pod $POD (namespace $NS)..."

kubectl cp "$ROOT/data/." "$NS/$POD:/app/data/" 2>/dev/null || echo "warn: data copy partial"
kubectl cp "$ROOT/artifacts/." "$NS/$POD:/app/artifacts/" 2>/dev/null || echo "warn: artifacts copy partial"
kubectl cp "$ROOT/models/." "$NS/$POD:/app/models/" 2>/dev/null || echo "warn: models copy partial"

echo "Done. Verify: kubectl exec -n $NS $POD -- ls /app/data"
