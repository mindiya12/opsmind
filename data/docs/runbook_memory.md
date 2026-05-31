# Runbook: Memory Pressure and JVM Garbage Collection Issues

## Overview
Memory pressure occurs when a JVM process (or any application) uses RAM faster than
the garbage collector can reclaim it. This leads to GC pauses (application freezes),
degraded response times, and ultimately OutOfMemoryError if unresolved.

## Symptoms
- Logs containing "GC major collection", "GC full collection", "OutOfMemoryError"
- Heap usage logs showing percentage approaching 90-95%
- Application response time spikes coinciding with GC events (GC pauses freeze the JVM)
- System memory available dropping below 1-2 GB
- Swap usage increasing

## JVM GC Pause Impact on Latency
When full GC runs, the JVM STOPS ALL APPLICATION THREADS ("stop the world").
A 3-second GC pause means ALL requests arriving during those 3 seconds time out.
This is why memory spikes cause sudden latency spikes even when the service is "up".

## Diagnosis Steps

### Step 1: Check current heap usage
```bash
# If using Java — connect to JVM and get heap stats
jcmd <pid> VM.native_memory
jstat -gcutil <pid> 1000 5  # GC stats every 1 second, 5 times
```

### Step 2: Check system memory
```bash
free -h                          # Available RAM and swap
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|SwapUsed"
ps aux --sort=-%mem | head -10  # Top memory consumers
```

### Step 3: Identify if a batch job triggered the spike
Look in application logs for: batch job started, bulk processing, report generation,
data export operations. These often cause sudden heap usage spikes.

### Step 4: Check GC log for frequency and duration
```bash
grep "GC" /var/log/app/analytics-service.log | tail -50
# Look for: frequency increasing, pause times growing, heap not recovering after GC
```

## Resolution

### Immediate: Pause the offending batch job
If a known batch job is causing the spike, pause it:
```bash
# Signal the analytics service to pause batch processing
curl -X POST http://localhost:8083/admin/batch/pause

# Or if it's a scheduled job
systemctl stop analytics-batch.timer
```
This allows GC to reclaim heap and prevents OutOfMemoryError.

### Immediate: Force GC (use with caution — causes a pause)
```bash
jcmd <pid> GC.run
```

### If OOM is imminent — restart the service
A controlled restart is better than an OOM crash:
```bash
systemctl restart analytics-service
# Service will come up with clean heap
```

### Long-term: Tune JVM heap settings
In the service's startup configuration:
```bash
# Increase max heap if the workload genuinely requires more memory
-Xmx8g          # Maximum heap (was 4g)
-Xms4g          # Initial heap (pre-allocate to avoid resize pauses)

# Use G1GC for large heaps — better at keeping GC pauses short
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200  # Target max pause of 200ms
```

### Long-term: Process batch jobs off-peak
Schedule memory-intensive batch jobs (reports, data exports) during low-traffic hours
(e.g., 02:00 AM) when the impact of GC pauses is minimal.
See: /etc/cron.d/analytics-jobs

## Disk Space Issues (Related)

### Symptoms
- System logs: "disk usage on /var reached X%" 
- Log rotation not keeping up with write rate
- Docker images consuming /var/lib/docker

### Immediate resolution
```bash
# Check what's consuming disk
du -sh /var/* | sort -rh | head -10
du -sh /var/log/* | sort -rh | head -10

# Force log rotation immediately
logrotate -f /etc/logrotate.d/nginx
logrotate -f /etc/logrotate.d/postgresql

# Clean old Docker images and containers
docker system prune -f         # Remove stopped containers, unused images
docker image prune -a -f       # Remove ALL unused images (free up GBs)

# Clean old WAL files (if DB archive is bloated)
# Only do this after confirming backups are current
psql -U admin -c "SELECT pg_switch_wal();"
```

### Prevention
- Log rotation configured to run daily, max 7 days retention for access logs
- Alert at 80% disk usage, critical at 90%
- Docker cleanup cron job: `docker system prune -f` weekly
- Monitor /var/lib/docker separately — images can grow unexpectedly after deployments