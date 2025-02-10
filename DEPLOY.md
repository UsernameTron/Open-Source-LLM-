# Sentiment Analysis API Deployment Guide

## Prerequisites

- Python 3.10 or higher
- Supervisor for process management
- Virtual environment

## Installation

1. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create .env file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

### Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of Gunicorn workers (default: CPU cores * 2 + 1)
- `LOG_LEVEL`: Logging level (default: INFO)
- `API_KEY`: API key for authentication
- `CORS_ORIGINS`: Comma-separated list of allowed origins

### Security

1. Generate a secure API key:
```bash
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

2. Update CORS origins in config.py for your domains

## Production Deployment

1. Set up Supervisor:
```bash
sudo cp supervisor.conf /etc/supervisor/conf.d/sentiment-analysis-api.conf
sudo supervisorctl reread
sudo supervisorctl update
```

2. Start the service:
```bash
sudo supervisorctl start sentiment-analysis-api
```

3. Check status:
```bash
sudo supervisorctl status sentiment-analysis-api
```

## Health Check

Test the deployment:
```bash
curl http://localhost:8000/api/health
```

## Monitoring

- Access logs: /var/log/supervisor/sentiment-analysis-api.log
- Error logs: /var/log/supervisor/sentiment-analysis-api.error.log
- Application metrics: http://localhost:8000/metrics

## Backup and Recovery

1. Regular backups:
```bash
# Backup configuration
cp .env .env.backup
cp supervisor.conf supervisor.conf.backup
```

2. Recovery:
```bash
# Restore configuration
cp .env.backup .env
cp supervisor.conf.backup supervisor.conf
sudo supervisorctl restart sentiment-analysis-api
```

## Troubleshooting

1. Check logs:
```bash
tail -f /var/log/supervisor/sentiment-analysis-api.log
```

2. Restart service:
```bash
sudo supervisorctl restart sentiment-analysis-api
```

3. Common issues:
- Port already in use: Check `netstat -tulpn | grep 8000`
- Permission errors: Check log files and directory permissions
- Memory issues: Monitor with `top` or `htop`

## Security Best Practices

1. Always use HTTPS in production
2. Regularly update dependencies
3. Monitor system resources
4. Implement rate limiting
5. Use API key authentication
6. Regular security audits
7. Keep backups
8. Monitor logs for suspicious activity

## Scaling

For higher load:
1. Increase number of workers in gunicorn.conf.py
2. Consider using load balancer
3. Implement caching if needed
4. Monitor performance metrics

## Support

For issues:
1. Check logs
2. Review configuration
3. Test health endpoint
4. Contact support team
