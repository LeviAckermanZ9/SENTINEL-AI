# ============================================
# SENTINEL-AI — Terraform Outputs
# ============================================

output "ec2_public_ip" {
  description = "Public IP of the SENTINEL-AI EC2 instance"
  value       = module.compute.ec2_public_ip
}

output "ec2_public_dns" {
  description = "Public DNS of the EC2 instance"
  value       = module.compute.ec2_public_dns
}

output "s3_model_bucket" {
  description = "S3 bucket for model artifacts"
  value       = module.storage.model_bucket_name
}

output "lambda_function_name" {
  description = "Lambda alert dispatcher function name"
  value       = module.compute.lambda_function_name
}

output "api_url" {
  description = "SENTINEL-AI API endpoint URL"
  value       = "http://${module.compute.ec2_public_dns}:8000"
}
