output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.claude_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.claude_instance.public_ip
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh -i your-private-key.pem ubuntu@${aws_instance.claude_instance.public_ip}"
}

output "tunnel_command" {
  description = "Command to create SSH tunnel for accessing the Claude interface"
  value       = "ssh -L 8080:localhost:8080 -i your-private-key.pem ubuntu@${aws_instance.claude_instance.public_ip}"
}

output "browser_url" {
  description = "URL to access Claude interface after creating the SSH tunnel"
  value       = "http://localhost:8080"
} 