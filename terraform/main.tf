# Phase 7 — zero-cost local IaC demo (Docker Desktop Kubernetes + Helm).
# Deploys SAINT-OPS to the cluster enabled in Docker Desktop → Settings → Kubernetes.
# No AWS/Azure resources are created.

provider "kubernetes" {
  config_path    = pathexpand(var.kubeconfig_path)
  config_context = var.kubeconfig_context
}

provider "helm" {
  kubernetes {
    config_path    = pathexpand(var.kubeconfig_path)
    config_context = var.kubeconfig_context
  }
}

resource "kubernetes_namespace" "saint" {
  metadata {
    name = var.namespace
  }
}

resource "helm_release" "saint_ops" {
  name             = var.release_name
  chart            = "${path.module}/../helm/saint-ops"
  namespace        = kubernetes_namespace.saint.metadata[0].name
  create_namespace = false
  timeout          = 600

  values = [
    file("${path.module}/../${var.helm_values_file}"),
    yamlencode({
      backend = {
        replicaCount = 1
        image = {
          pullPolicy = "IfNotPresent"
        }
      }
      frontend = {
        replicaCount = 1
      }
      autoscaling = {
        enabled     = true
        minReplicas = 1
        maxReplicas = 3
      }
      ingress = {
        enabled = false
      }
    }),
  ]

  depends_on = [kubernetes_namespace.saint]
}
