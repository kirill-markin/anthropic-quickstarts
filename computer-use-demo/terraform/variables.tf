variable "aws_region" {
  description = "AWS region to deploy the infrastructure"
  type        = string
  default     = "us-west-2"
}

variable "aws_profile" {
  description = "AWS CLI profile to use for authentication"
  type        = string
  default     = "default"
}

variable "ami_id" {
  description = "AMI ID for Ubuntu 22.04"
  type        = string
  default     = "ami-0ee5b2a8d3fa27713" # Ubuntu 22.04 LTS in us-west-2, update accordingly for other regions
}

variable "ssh_public_key" {
  description = "SSH public key for EC2 instance access"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key for Claude"
  type        = string
  sensitive   = true
} 