#!/bin/bash
set -e

# Update system packages
apt-get update
apt-get upgrade -y

# Install required dependencies
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    unzip

# Add Docker's official GPG key
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add ubuntu user to docker group
usermod -aG docker ubuntu

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Create directory for Anthropic settings
mkdir -p /home/ubuntu/.anthropic
chown -R ubuntu:ubuntu /home/ubuntu/.anthropic

# Create the container startup script
cat > /home/ubuntu/start-claude.sh <<'EOF'
#!/bin/bash

export ANTHROPIC_API_KEY=${anthropic_api_key}

# Run the Claude container
# Note: Using -d instead of -it for server deployment (runs in background)
# Added --name for better server-side container management
# IMPORTANT: The --restart unless-stopped flag is removed for debugging purposes
# For production use, add the following line before --name claude:
#     --restart unless-stopped \
docker run \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    -v /home/ubuntu/.anthropic:/home/computeruse/.anthropic \
    -p 5900:5900 \
    -p 8501:8501 \
    -p 6080:6080 \
    -p 8080:8080 \
    -e WIDTH=1920 \
    -e HEIGHT=1080 \
    -d \
    --name claude \
    ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
EOF

chmod +x /home/ubuntu/start-claude.sh
chown ubuntu:ubuntu /home/ubuntu/start-claude.sh

# Set up systemd service for automatic start
cat > /etc/systemd/system/claude.service <<EOF
[Unit]
Description=Claude Computer Use Demo
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ubuntu
ExecStart=/home/ubuntu/start-claude.sh
ExecStop=docker stop claude
ExecStopPost=docker rm claude

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the Claude service
systemctl enable claude.service
systemctl start claude.service

echo "Claude Computer Use Demo setup complete" 