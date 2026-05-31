# Runbook: Database Connection Pool Exhaustion

## Overview
Connection pool exhaustion occurs when all available database connections are in use
and new requests cannot acquire a connection. This typically causes cascading failures
across all services that depend on the database.

## Symptoms
- Error logs containing "connection pool exhausted" or "max_connections reached"
- HTTP 502 or 503 errors from services that query the database
- Application logs showing "Failed to acquire DB connection from pool"
- Database metrics showing connections/max_connections ratio > 90%
- Nginx upstream timeout errors coinciding with pool exhaustion

## Immediate Diagnosis Steps

### Step 1: Confirm pool exhaustion
```bash
# Check current connection count vs maximum
psql -U admin -c "SELECT count(*), max_conn FROM pg_stat_activity, (SELECT setting::int AS max_conn FROM pg_settings WHERE name='max_connections') AS s;"

# Identify who is holding connections
psql -U admin -c "SELECT pid, usename, application_name, state, wait_event, query_start, query FROM pg_stat_activity ORDER BY query_start;"
```

### Step 2: Identify long-running or idle connections
```bash
# Find connections idle for more than 5 minutes
psql -U admin -c "SELECT pid, usename, state, now() - state_change AS idle_duration FROM pg_stat_activity WHERE state = 'idle' AND now() - state_change > interval '5 minutes';"
```

### Step 3: Check for connection leaks in application logs
Look for: "connection not returned to pool", "connection leak detected", services that
open connections but do not close them after requests complete.

## Resolution

### Immediate fix (terminate stale connections)
```bash
# Terminate all idle connections older than 5 minutes (SAFE — only idle ones)
psql -U admin -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND now() - state_change > interval '5 minutes';"
```

### Short-term fix (increase pool size temporarily)
Edit /etc/postgresql/postgresql.conf:
```
max_connections = 100  # increase from default 50 — requires restart
```
Restart PostgreSQL: `systemctl restart postgresql`

Note: This is a temporary measure. High max_connections increases shared_buffers
consumption and can degrade overall DB performance.

### Long-term fixes
1. Implement connection pooling middleware (PgBouncer) between app and DB
2. Review application code for connection leaks — ensure all connections are
   released in finally blocks or use connection context managers
3. Set connection timeout in app config so abandoned connections auto-expire
4. Configure pgbouncer pool_mode=transaction for efficient connection sharing

## Prevention
- Set `idle_in_transaction_session_timeout = 300000` in postgresql.conf (5 min)
- Monitor pool utilization — alert at 70%, critical at 85%
- Run `pg_stat_activity` checks in your monitoring dashboard
- Implement circuit breaker in application to fail fast when pool is exhausted
  rather than queuing requests that will all time out

## Related Incidents
- 2024-01-15 03:42: Pool exhausted during overnight analytics job. 
  Root cause: long-running reports held connections open while processing.
  Fix: analytics jobs now use a separate, dedicated connection pool.

## Escalation
If pool exhaustion recurs within 24 hours after above steps, escalate to DBA team.
This indicates a deeper connection leak that requires code-level investigation.