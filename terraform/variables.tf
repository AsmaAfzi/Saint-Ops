variable "kubeconfig_path" {
  description = "Path to kubeconfig (Docker Desktop writes ~/.kube/config)"
  type        = string
  default     = "~/.kube/config"
}

variable "kubeconfig_context" {
  description = "kubectl context — use docker-desktop for Docker Desktop Kubernetes"
  type        = string
  default     = "docker-desktop"
}

variable "release_name" {
  description = "Helm release name"
  type        = string
  default     = "saint-dev"
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "saint-dev"
}

variable "helm_values_file" {
  description = "Helm values file relative to repo root"
  type        = string
  default     = "helm/saint-ops/values-dev.yaml"
}
