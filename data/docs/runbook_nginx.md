# Runbook: Nginx Upstream Timeout and 502/503 Errors

## Overview
Nginx upstream errors occur when the backend servers (application services) fail to
respond within the configured timeout period, or are completely unavailable.
These manifest as HTTP 502 Bad Gateway or 503 Service Unavailable responses.

## Symptoms
- Nginx access logs: "upstream timed out (110: Connection timed out)"
- Nginx error logs: "connect() failed (111: Connection refused)"
- HTTP 502 errors visible to end users
- High upstream error rate in nginx metrics (>5% is concerning, >20% is critical)
- Upstream response time p99 exceeding timeout threshold (default: 30000ms)

## Common Root Causes

### 1. Backend service is down
The application service nginx is proxying to has crashed or is not running.

### 2. Backend is overloaded
The service is running but cannot process requests fast enough — often caused by:
- Database connection pool exhaustion (see runbook_database.md)
- JVM garbage collection pauses causing application freezes
- CPU or memory saturation on the backend host

### 3. Network issue between nginx and backend
Firewall rule change, network partition, or DNS resolution failure.

### 4. Nginx timeout misconfiguration
Timeout set too low for the expected response time of the backend.

## Diagnosis Steps

### Step 1: Check if backends are responding
```bash
# Test the upstream directly (bypass nginx)
curl -v http://localhost:8080/health  # auth-service
curl -v http://localhost:8081/health  # orders-service
curl -v http://localhost:8082/health  # users-service
```

### Step 2: Check nginx upstream error rate
```bash
# Count 502s in the last 100 lines of access log
tail -n 100 /var/log/nginx/access.log | grep ' 502 ' | wc -l

# Check error log for specific upstream error messages
tail -n 50 /var/log/nginx/error.log
```

### Step 3: Check backend service status
```bash
systemctl status orders-service users-service auth-service
journalctl -u orders-service --since "10 minutes ago"
```

### Step 4: Examine backend resource usage
```bash
top -b -n1 | head -20       # CPU and memory
free -h                      # Available memory
iostat -x 1 3               # Disk I/O (if latency seems I/O-bound)
```

## Resolution

### If backend is down — restart it
```bash
systemctl restart orders-service  # or whichever service is affected
systemctl status orders-service   # confirm it came up
```

### If backend is overloaded due to DB issues
Follow runbook_database.md first. Once DB is healthy, nginx errors will resolve
automatically as backends recover.

### If GC pauses are causing timeouts
Increase nginx proxy_read_timeout temporarily:
```nginx
location /api/ {
    proxy_read_timeout 60s;  # increase from 30s
    proxy_connect_timeout 10s;
}
```
Reload nginx: `nginx -s reload` (no downtime)

Then investigate JVM heap — see runbook_memory.md.

### Emergency: Enable maintenance page
If backend is unrecoverable and you need to prevent further 502s:
```bash
# Serve a maintenance page while fixing the backend
cp /etc/nginx/snippets/maintenance.conf /etc/nginx/conf.d/
nginx -s reload
```

## Circuit Breaker Configuration
Nginx can be configured to automatically stop sending traffic to a failing upstream:
```nginx
upstream backend {
    server 127.0.0.1:8080;
    max_fails=3;          # Mark as failed after 3 failures
    fail_timeout=30s;     # Stop sending to it for 30 seconds
}
```

## Prevention
- Health check endpoints on all services, polled by nginx every 5s
- Alerting when upstream error rate exceeds 5% for 2 consecutive minutes
- Canary deployments to catch backend issues before they affect 100% of traffic
- Rate limiting on endpoints prone to causing backend overload