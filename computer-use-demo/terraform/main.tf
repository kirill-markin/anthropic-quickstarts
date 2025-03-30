terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

resource "aws_vpc" "claude_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "claude-vpc"
  }
}

resource "aws_subnet" "claude_subnet" {
  vpc_id                  = aws_vpc.claude_vpc.id
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = true
  availability_zone       = "${var.aws_region}a"

  tags = {
    Name = "claude-subnet"
  }
}

resource "aws_internet_gateway" "claude_igw" {
  vpc_id = aws_vpc.claude_vpc.id

  tags = {
    Name = "claude-igw"
  }
}

resource "aws_route_table" "claude_route_table" {
  vpc_id = aws_vpc.claude_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.claude_igw.id
  }

  tags = {
    Name = "claude-route-table"
  }
}

resource "aws_route_table_association" "claude_rta" {
  subnet_id      = aws_subnet.claude_subnet.id
  route_table_id = aws_route_table.claude_route_table.id
}

resource "aws_security_group" "claude_sg" {
  name        = "claude-security-group"
  description = "Security group for Claude Computer Use Demo"
  vpc_id      = aws_vpc.claude_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Claude interface"
  }

  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Streamlit"
  }

  ingress {
    from_port   = 5900
    to_port     = 5900
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "VNC"
  }

  ingress {
    from_port   = 6080
    to_port     = 6080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "noVNC"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "claude-security-group"
  }
}

resource "aws_key_pair" "claude_key_pair" {
  key_name   = "claude-key"
  public_key = var.ssh_public_key
}

resource "aws_instance" "claude_instance" {
  ami                    = var.ami_id
  instance_type          = "t3.2xlarge"
  key_name               = aws_key_pair.claude_key_pair.key_name
  vpc_security_group_ids = [aws_security_group.claude_sg.id]
  subnet_id              = aws_subnet.claude_subnet.id

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/aws.sh", {
    anthropic_api_key = var.anthropic_api_key
  })

  tags = {
    Name = "claude-computer-use-demo"
  }
}
