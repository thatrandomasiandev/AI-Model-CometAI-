# CometAI Production Deployment Guide

This guide will help you deploy CometAI as a production web service that runs independently and stays online.

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Clone and prepare:**
   ```bash
   git clone <your-repo>
   cd cometai
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Check status:**
   ```bash
   curl http://localhost:8080/health
   ```

### Option 2: Direct Server Deployment

1. **Run the deployment script:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

2. **The script will:**
   - Create a system user for the service
   - Install dependencies
   - Set up systemd service
   - Configure firewall
   - Start the service

## 🌐 Cloud Platform Deployment

### AWS EC2

1. **Launch an EC2 instance:**
   - Instance type: `m5.2xlarge` or larger (8+ vCPUs, 32+ GB RAM)
   - AMI: Ubuntu 22.04 LTS
   - Security group: Allow inbound port 8080

2. **Connect and deploy:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   git clone <your-repo>
   cd cometai
   ./deploy.sh
   ```

3. **Configure domain (optional):**
   - Point your domain to the EC2 public IP
   - Update `CORS_ORIGINS` in environment variables

### Google Cloud Platform

1. **Create a Compute Engine instance:**
   ```bash
   gcloud compute instances create cometai-server \
     --machine-type=n1-standard-8 \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=100GB \
     --tags=http-server
   ```

2. **Create firewall rule:**
   ```bash
   gcloud compute firewall-rules create allow-cometai \
     --allow tcp:8080 \
     --source-ranges 0.0.0.0/0 \
     --target-tags http-server
   ```

3. **SSH and deploy:**
   ```bash
   gcloud compute ssh cometai-server
   # Then follow the deployment steps
   ```

### DigitalOcean Droplet

1. **Create a droplet:**
   - Size: 8 GB RAM / 4 vCPUs or larger
   - Image: Ubuntu 22.04 LTS
   - Add your SSH key

2. **Deploy:**
   ```bash
   ssh root@your-droplet-ip
   git clone <your-repo>
   cd cometai
   ./deploy.sh
   ```

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```bash
cp env.example .env
nano .env
```

Key settings:
- `HOST=0.0.0.0` - Bind to all interfaces
- `PORT=8080` - Server port
- `ENVIRONMENT=production` - Production mode
- `CORS_ORIGINS=*` - Allowed origins (restrict in production)

### Model Configuration

Edit `config.yaml` to customize:
- Model selection
- Generation parameters
- Performance settings
- Memory optimization

## 📊 Monitoring & Management

### Service Management (systemd)

```bash
# Check status
sudo systemctl status cometai

# Start/stop/restart
sudo systemctl start cometai
sudo systemctl stop cometai
sudo systemctl restart cometai

# View logs
sudo journalctl -u cometai -f
```

### Docker Management

```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Update and restart
docker-compose pull
docker-compose up -d
```

### Health Monitoring

The server provides several monitoring endpoints:

- `GET /health` - Health check with system info
- `GET /api/model/info` - Model status and information
- `GET /` - Service overview

Example health check script:
```bash
#!/bin/bash
if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ CometAI is healthy"
else
    echo "❌ CometAI is down"
    # Restart service
    sudo systemctl restart cometai
fi
```

## 🔒 Security Considerations

### Production Security

1. **Firewall Configuration:**
   ```bash
   # Only allow necessary ports
   sudo ufw enable
   sudo ufw allow ssh
   sudo ufw allow 8080
   ```

2. **Reverse Proxy (Nginx):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8080;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **SSL/TLS:**
   - Use Let's Encrypt for free SSL certificates
   - Configure HTTPS redirect

4. **API Security:**
   - Implement API key authentication
   - Rate limiting
   - Input validation

### Resource Limits

The service is configured with resource limits:
- Memory: 20GB max (adjust based on your model)
- File descriptors: 65536
- Process limits via systemd

## 🚨 Troubleshooting

### Common Issues

1. **Model fails to load:**
   - Check available memory (16GB+ recommended)
   - Verify model files are accessible
   - Check logs: `sudo journalctl -u cometai -f`

2. **Service won't start:**
   - Check port availability: `sudo netstat -tlnp | grep 8080`
   - Verify permissions: `ls -la /opt/cometai`
   - Check systemd status: `sudo systemctl status cometai`

3. **High memory usage:**
   - Enable quantization in `config.yaml`
   - Reduce `max_tokens` setting
   - Monitor with `htop` or `free -h`

4. **Slow responses:**
   - Check CPU usage
   - Verify GPU acceleration (if available)
   - Adjust generation parameters

### Log Locations

- **Systemd service:** `sudo journalctl -u cometai`
- **Application logs:** `/opt/cometai/logs/cometai_server.log`
- **Docker logs:** `docker-compose logs`

## 📈 Performance Optimization

### Hardware Recommendations

- **CPU:** 8+ cores (Intel Xeon, AMD EPYC, or Apple M-series)
- **RAM:** 32GB+ (16GB minimum)
- **Storage:** SSD with 100GB+ free space
- **GPU:** Optional but recommended (NVIDIA with CUDA support)

### Software Optimizations

1. **Enable quantization:**
   ```yaml
   # config.yaml
   performance:
     load_in_8bit: true
   ```

2. **Optimize generation settings:**
   ```yaml
   model:
     max_tokens: 512  # Reduce for faster responses
     temperature: 0.3  # Lower for more focused output
   ```

3. **System tuning:**
   ```bash
   # Increase file limits
   echo "* soft nofile 65536" >> /etc/security/limits.conf
   echo "* hard nofile 65536" >> /etc/security/limits.conf
   ```

## 🔄 Updates & Maintenance

### Updating the Service

1. **Pull latest code:**
   ```bash
   cd /opt/cometai
   sudo -u cometai git pull
   ```

2. **Update dependencies:**
   ```bash
   sudo -u cometai /opt/cometai/venv/bin/pip install -r requirements.txt
   ```

3. **Restart service:**
   ```bash
   sudo systemctl restart cometai
   ```

### Backup & Recovery

1. **Backup configuration:**
   ```bash
   tar -czf cometai-backup.tar.gz /opt/cometai/config.yaml /opt/cometai/logs
   ```

2. **Model cache backup:**
   ```bash
   tar -czf model-cache.tar.gz /opt/cometai/.cache
   ```

## 📞 Support

For issues and questions:
1. Check the logs first
2. Review this deployment guide
3. Check system resources
4. Verify network connectivity

The server is designed to be robust and self-healing, with automatic restarts and graceful error handling.
