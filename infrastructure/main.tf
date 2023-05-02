terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "3.49.0"
    }
  }
}

#Resource to generate random pet names
resource "random_pet" "ray_ask_pet_name" {
  length    = 1
}

# Configure the Azure Provider
provider "azurerm" {
  features {

  }
}

resource "azurerm_resource_group" "ray_aks_rg" {
  location = "East US"
  name     = "${random_pet.ray_ask_pet_name.id}-aks-rg"
  tags = {
    environment = "dev"
    Terraform   = "true"
  }
}

resource "azurerm_virtual_network" "ray_aks_vnet" {
  address_space       = ["10.0.0.0/16"]
  location            = azurerm_resource_group.ray_aks_rg.location
  name                = "${random_pet.ray_ask_pet_name.id}-aks-vnet"
  resource_group_name = azurerm_resource_group.ray_aks_rg.name
}

resource "azurerm_subnet" "ray_aks_subnet" {
  address_prefixes         = ["10.0.1.0/24"]
  name                     = "${random_pet.ray_ask_pet_name.id}-aks-subnet"
  resource_group_name      = azurerm_resource_group.ray_aks_rg.name
  virtual_network_name     = azurerm_virtual_network.ray_aks_vnet.name
}

resource "azurerm_kubernetes_cluster" "ray_aks_cluster" {
  location            = azurerm_resource_group.ray_aks_rg.location
  name                = "${random_pet.ray_ask_pet_name.id}-aks-cluster"
  resource_group_name = azurerm_resource_group.ray_aks_rg.name
  dns_prefix          = "${random_pet.ray_ask_pet_name.id}-aks-dns"

  default_node_pool {
    name    = "aksnodepool"
    node_count = 1
    vnet_subnet_id = azurerm_subnet.ray_aks_subnet.id
    vm_size = "Standard_D4s_v3"
    temporary_name_for_rotation = true
  }
  identity {
    type = "SystemAssigned"
  }
  network_profile {
    network_plugin = "kubenet"
    service_cidr = "10.0.2.0/24"
    dns_service_ip = "10.0.2.10"
    load_balancer_sku = "standard"
  }
    depends_on = [azurerm_subnet.ray_aks_subnet]
}