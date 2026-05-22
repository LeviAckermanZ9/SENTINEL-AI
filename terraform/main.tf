# ============================================
# SENTINEL-AI — Terraform Main Configuration
# ============================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # S3 remote state backend
  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "aws" {
  region                      = var.aws_region
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true
  access_key                  = "AKIAIOSFODNN7EXAMPLE"
  secret_key                  = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

  default_tags {
    tags = {
      Project     = "SENTINEL-AI"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# --- Modules ---

module "networking" {
  source      = "./modules/networking"
  environment = var.environment
  project     = var.project_name
}

module "storage" {
  source      = "./modules/storage"
  environment = var.environment
  project     = var.project_name
}

module "compute" {
  source            = "./modules/compute"
  environment       = var.environment
  project           = var.project_name
  vpc_id            = module.networking.vpc_id
  public_subnet_id  = module.networking.public_subnet_id
  security_group_id = module.networking.security_group_id
  s3_bucket_arn     = module.storage.model_bucket_arn
  instance_type     = var.instance_type
  slack_webhook_url = var.slack_webhook_url
}
