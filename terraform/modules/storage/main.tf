# ============================================
# SENTINEL-AI — Storage Module
# ============================================

variable "environment" {
  type = string
}

variable "project" {
  type = string
}

# --- S3 Bucket for Model Artifacts ---
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "${var.project}-model-artifacts-${var.environment}"

  tags = {
    Name = "${var.project}-model-artifacts"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Lifecycle policy for old versions
resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id

  rule {
    id     = "cleanup-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# --- Outputs ---
output "model_bucket_name" {
  value = aws_s3_bucket.model_artifacts.bucket
}

output "model_bucket_arn" {
  value = aws_s3_bucket.model_artifacts.arn
}
