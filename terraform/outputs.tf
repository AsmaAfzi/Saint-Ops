output "kubeconfig_context" {
  value       = var.kubeconfig_context
  description = "kubectl context used for deploy"
}

output "namespace" {
  value       = var.namespace
  description = "Kubernetes namespace"
}

output "helm_release" {
  value       = helm_release.saint_ops.name
  description = "Deployed Helm release"
}

output "demo_commands" {
  value       = <<-EOT
    kubectl config use-context ${var.kubeconfig_context}
    kubectl get pods -n ${var.namespace}
    kubectl apply -f scripts/k8s_seed_pod.yaml
    kubectl wait --for=condition=Ready pod/seed-pvc -n ${var.namespace} --timeout=120s
    kubectl cp ./data/. ${var.namespace}/seed-pvc:/mnt/data/
    kubectl cp ./artifacts/. ${var.namespace}/seed-pvc:/mnt/artifacts/
    kubectl cp ./models/. ${var.namespace}/seed-pvc:/mnt/models/
    kubectl delete pod seed-pvc -n ${var.namespace}
    kubectl port-forward -n ${var.namespace} svc/${var.release_name}-saint-ops-frontend 3000:80
    kubectl port-forward -n ${var.namespace} svc/${var.release_name}-saint-ops-backend 8000:8000
  EOT
  description = "Post-apply demo commands"
}
