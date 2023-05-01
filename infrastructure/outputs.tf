output "resource_group_name" {
  value = azurerm_resource_group.ray_aks_rg.name
  description = "The name of the resource group"
}

output "ray_aks_cluster_cluster_name" {
  value = azurerm_kubernetes_cluster.ray_aks_cluster.name
  description = "The name of the AKS cluster"
}

output "ray_aks_cluster_kube_config" {
  value = azurerm_kubernetes_cluster.ray_aks_cluster.kube_config_raw
  description = "The Kubernetes configuration file for the AKS cluster"
  sensitive   = true
}

output "client_certificate" {
  value     = azurerm_kubernetes_cluster.ray_aks_cluster.kube_config.0.client_certificate
  description = "The client certificate for the AKS cluster"
  sensitive = true
}

output "host" {
  value = azurerm_kubernetes_cluster.ray_aks_cluster.kube_config.0.host
  description = "The host for the AKS cluster"
  sensitive = true
}