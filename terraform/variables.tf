# ============================================
# SENTINEL-AI — Terraform Variables
# ============================================

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name for resource tagging"
  type        = string
  default     = "sentinel-ai"
}

variable "instance_type" {
  description = "EC2 instance type (free tier: t2.micro)"
  type        = string
  default     = "t2.micro"
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alert notifications"
  type        = string
  default     = ""
  sensitive   = true
}
