# Claude Computer Use Demo Terraform Module

This Terraform module deploys the Anthropic Claude Computer Use Demo on AWS EC2.

## Prerequisites

- [Terraform](https://www.terraform.io/downloads) installed (version 1.2.0 or newer)
- AWS CLI configured with appropriate credentials
- Anthropic API key

## Usage

1. Create a `terraform.tfvars` file based on the provided example:

```bash
cp terraform.tfvars.example terraform.tfvars
```

2. Edit the `terraform.tfvars` file to include your SSH public key and Anthropic API key.

3. Initialize Terraform:

```bash
terraform init
```

4. Preview the changes:

```bash
terraform plan
```

5. Apply the changes:

```bash
terraform apply
```

6. After successful deployment, use the outputs to connect to your EC2 instance:

```bash
# SSH directly to the instance
ssh -i your-private-key.pem ubuntu@<instance_public_ip>

# Create SSH tunnel for the Claude interface
ssh -L 8080:localhost:8080 -i your-private-key.pem ubuntu@<instance_public_ip>
```

7. Access the Claude interface at http://localhost:8080

## Notes

- The EC2 instance type is set to t3.large, which should be sufficient for running the Claude Computer Use Demo.
- The instance will have Docker pre-installed and the Claude container will start automatically on boot.
- Make sure your private key corresponds to the public key provided in terraform.tfvars.

## Cleanup

To destroy all created resources:

```bash
# Navigate to the terraform directory
cd computer-use-demo/terraform

# Run the destroy command
terraform destroy
```

This command will:
1. Show a plan of what resources will be destroyed
2. Ask for confirmation (type 'yes' to proceed)
3. Remove all AWS resources created by this Terraform configuration including:
   - EC2 instance
   - Security groups
   - Network resources
   - Any other associated components

Once completed, all resources and associated costs will be fully terminated.

## Cost Estimation

- t3.large on-demand: ~$60-75/month (running 24/7)
- With scheduled stopping during non-working hours: ~$20-30/month
- Additional costs for storage (~$3/month) and network traffic 