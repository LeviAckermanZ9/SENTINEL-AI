# ============================================
# SENTINEL-AI — Compute Module
# ============================================

variable "environment" {
  type = string
}

variable "project" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "public_subnet_id" {
  type = string
}

variable "security_group_id" {
  type = string
}

variable "s3_bucket_arn" {
  type = string
}

variable "instance_type" {
  type    = string
  default = "t2.micro"
}

variable "slack_webhook_url" {
  type      = string
  default   = ""
  sensitive = true
}

# --- AMI Data Source (Amazon Linux 2023) ---
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# --- IAM Role for EC2 ---
resource "aws_iam_role" "sentinel_ec2" {
  name = "${var.project}-ec2-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# S3 least-privilege policy
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project}-s3-access"
  role = aws_iam_role.sentinel_ec2.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          var.s3_bucket_arn,
          "${var.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "sentinel" {
  name = "${var.project}-instance-profile-${var.environment}"
  role = aws_iam_role.sentinel_ec2.name
}

# --- EC2 Instance ---
resource "aws_instance" "sentinel" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  subnet_id              = var.public_subnet_id
  vpc_security_group_ids = [var.security_group_id]
  iam_instance_profile   = aws_iam_instance_profile.sentinel.name

  user_data = <<-EOF
    #!/bin/bash
    set -e

    # Install Docker
    yum update -y
    yum install -y docker
    systemctl start docker
    systemctl enable docker
    usermod -aG docker ec2-user

    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Pull and run SENTINEL-AI
    docker pull ghcr.io/sentinel-ai/sentinel-api:latest
    docker run -d --name sentinel-api \
      -p 8000:8000 \
      --restart unless-stopped \
      ghcr.io/sentinel-ai/sentinel-api:latest
  EOF

  root_block_device {
    volume_size = 30
    volume_type = "gp3"
    encrypted   = true
  }

  tags = {
    Name = "${var.project}-server-${var.environment}"
  }
}

# --- IAM Role for Lambda ---
resource "aws_iam_role" "lambda_role" {
  name = "${var.project}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# --- Lambda Alert Dispatcher ---
data "archive_file" "lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/lambda_function.zip"

  source {
    content  = <<-PYTHON
    import json
    import os
    import urllib.request

    def lambda_handler(event, context):
        """Dispatch alerts to Slack when fake news rate exceeds threshold."""
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        threshold = float(os.environ.get("THRESHOLD_FAKE", "0.30"))

        fake_rate = event.get("fake_rate", 0)

        if fake_rate > threshold and webhook_url:
            message = {
                "text": f":warning: *SENTINEL-AI Alert*\nFake news detection rate: {fake_rate:.1%}\nThreshold: {threshold:.1%}\nAction required: Review recent classifications."
            }
            req = urllib.request.Request(
                webhook_url,
                data=json.dumps(message).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req)

        return {
            "statusCode": 200,
            "body": json.dumps({"alert_sent": fake_rate > threshold})
        }
    PYTHON
    filename = "lambda_function.py"
  }
}

resource "aws_lambda_function" "alert_dispatcher" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project}-alert-dispatcher"
  role             = aws_iam_role.lambda_role.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 30
  memory_size      = 128

  environment {
    variables = {
      SLACK_WEBHOOK_URL = var.slack_webhook_url
      THRESHOLD_FAKE    = "0.30"
    }
  }

  tags = {
    Name = "${var.project}-alert-dispatcher"
  }
}

# --- Outputs ---
output "ec2_public_ip" {
  value = aws_instance.sentinel.public_ip
}

output "ec2_public_dns" {
  value = aws_instance.sentinel.public_dns
}

output "lambda_function_name" {
  value = aws_lambda_function.alert_dispatcher.function_name
}
