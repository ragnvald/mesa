# Draft plan: Cloud processing for MESA

## Purpose
Enable users with low-capacity computers to offload the main vector processing workload to a Linux server with PostGIS, while keeping the local MESA workflow simple and familiar. Outputs include GeoParquet and MBTiles.

## Assumptions
- Server specs: 128 GB RAM, fast local SSD. GPU optional.
- Server OS: Ubuntu latest LTS.
- PostGIS is installed and available on the server.
- Job dispatch is sequential (single worker); requests are queued.
- Within a job, processing can be parallel (GPU-based and/or multi-core CPU), depending on the resource used (PostGIS/CPU/GPU). Some stages may run CPU parallelism alongside GPU work.
- Initial scope: only the main processing step (not projects, not lines).
- Users authenticate using a token issued at registration.
- Minimal upload payload: 100 MB GeoParquet tables, zipped into a single workload.
- Uploads are small to moderate (assets + settings). Outputs are small to moderate.
- There is an admin web console for tokens, queue management, and policy controls.

## User-facing story
1. User has imported data and configured processing parameters and indexes.
2. User clicks "Process in cloud" from the local UI.
3. A local dialog opens where the user enters name, email, project summary, and affiliation.
4. User presses Register. Server validates registration and sends a token by email.
5. Existing users are told a token was emailed. New users receive a new token.
6. User enters the token in the dialog to confirm.
7. User clicks "Upload and process" to upload the zipped workload and start the job.
8. If the user already has jobs, a table lists them with status and a Download action.
9. Downloads place GeoParquet and MBTiles in the standard local output folders.

## High-level architecture
- Web API (public): auth, job submission, upload URLs, status polling, download URLs.
- Processing worker (private): dequeues jobs, runs pipeline, writes outputs.
- Storage: object storage for uploads/outputs (S3-compatible) or local SSD with cleanup.
- Database: Postgres + PostGIS for processing and job metadata.
- Styling service: server-side styling via an internal API referenced by OpenAPI docs.
- Admin console (web UI): manage tokens, queue, blacklists, and priorities.

## API surface (draft)
- POST /register: email + policy checks -> token
- POST /auth/verify: token -> user profile, quotas
- POST /jobs: token + job metadata -> job id, upload URLs
- GET /jobs/{id}: job status, progress, logs preview
- POST /jobs/{id}/complete-upload: confirm all parts uploaded
- GET /jobs/{id}/results: download URLs
- POST /jobs/{id}/cancel: admin or owner

## Token model
- Token maps to: user id, email, created, status, priority, quotas.
- Token is stored by MESA locally (config.ini) and only sent via HTTPS.
- Tokens can be revoked, rotated, or blacklisted by email domain rules.

## Job model
- Job state: queued -> running -> finished | failed | cancelled.
- Job inputs: assets file(s), settings file, and minimal metadata.
- Job outputs: generated files, logs, and a manifest.json.
- Queue is ordered by priority, then created time.

## Processing pipeline (server side)
- Validate token and inputs.
- Load assets into a staging area (temporary schema or temp tables).
- Run the existing processing pipeline with server paths.
- Write outputs to a job-specific output folder.
- Apply server-side styling using the styling API for MBTiles outputs.
- Build manifest with checksums and sizes.
- Clean up staging tables after success or failure.

## PostGIS vs Python vector stack evaluation (time-to-result)
Goal: maximize speed. Energy cost is not considered.

### Guiding principle
- Prefer PostGIS for set-based spatial joins, indexing, and topology-heavy operations.
- Prefer Python/vector libraries when operations are simpler in code or need custom logic.
- Use the metric: total wall-clock time for a full job, not micro-benchmarks alone.

### Likely PostGIS wins
- Spatial joins with indexing (ST_Intersects, ST_Within) when tables are large and indexed.
- Complex polygon operations where robust topology matters (ST_Union, ST_Intersection, ST_Difference) if GPU kernel parity is weak.
- Attribute aggregation + spatial grouping in a single SQL pass.
- Any operation that benefits from query planning, statistics, and spatial indexes.

### Likely Python/vector wins
- Small or medium datasets where the overhead of database staging dominates.
- Custom rules that are easier to express in code than SQL.
- Pipeline steps that already exist as Python helpers and are stable.

### Decision checklist (per pipeline step)
- Is the operation a set-based spatial join or topology-heavy geometry op? If yes, try PostGIS.
- Can the step be expressed in SQL with spatial indexes? If yes, PostGIS first.
- Is the step custom or hard to express in SQL? If yes, keep it in Python.
- Is the dataset small enough that DB staging is overhead? If yes, consider Python.

### Evaluation plan
- Create a benchmark dataset that matches the minimal payload (100 MB GeoParquet).
- For each pipeline step, compare:
	- Pure PostGIS implementation.
	- Python/vector implementation.
	- Hybrid (PostGIS for indexing/join, Python for custom transforms).
- Record per-step time, end-to-end time, and data transfer overhead.
- Pick the fastest per step, then re-run the full pipeline to confirm no regressions.

## Data transfer plan
- Use pre-signed uploads to reduce API load.
- MESA uploads to storage, then calls complete-upload.
- Client creates a zipped workload locally before upload.
- MESA downloads outputs using signed URLs.
- Optional compression for outputs if large (zip or tar.gz).

## Retention, notifications, and download windows
- Uploaded data is deleted after download or after 48 hours (whichever comes first).
- The user receives an email when results are ready to download.
- Downloads are time-limited (default 48 hours).
- Admin can set a per-user lag/retention window; the admin-defined default may be longer than 48 hours.

## Admin console features
- View queue, job details, and live logs tail.
- Cancel or delete queued/running jobs.
- Manage tokens (create, revoke, rotate).
- Blacklist by email or domain.
- Set per-user priority and quotas.

## Security and compliance
- HTTPS only; no token in URLs.
- Token stored encrypted at rest in server database.
- Use per-job storage paths with random IDs.
- Hard limits: max upload size, max job duration, max storage per user.
- Audit log for admin actions and job access.

## Server provisioning and environment discovery
Goal: define what happens when a new bare-metal server is installed, and how the system learns the local environment (CPU, RAM, disk, PostGIS, GPU, drivers).

### Provisioning steps (operator runbook)
1. Provision a bare-metal host with Ubuntu LTS and a dedicated data disk.
2. Install system packages: Python 3.11, the newest compatible Postgres + PostGIS, build tools, and NVIDIA GPU drivers (GPU required).
3. Create a dedicated service account and a fixed install root (e.g., /opt/mesa-server).
4. Create a Python venv in the install root and install the pinned requirements.
5. Deploy server config (env vars, secrets, ports, storage paths).
6. Initialize database schema (users, tokens, jobs, audit logs).
7. Run environment probe (see below) and store results.
8. Start services (API + worker) and verify health endpoints.

### Environment probe (bootstrap)
The server must collect and persist key environment details on first run and on demand:
- OS version, kernel, CPU model and cores, RAM total and free.
- Disk mount(s), free space, filesystem type, and IO health.
- Postgres + PostGIS versions and extension status.
- NVIDIA GPU model, driver version, CUDA runtime version (GPU required).
- Python runtime version and key package versions.

Store the probe results in the database (server_info table) and expose a read-only endpoint for admin diagnostics.
The worker should re-check disk space before starting jobs and refuse work below a threshold.

### Install layout and lifecycle
- Keep everything under a single root folder (code, venv, logs, configs).
- Allow rolling updates by deploying a new version beside the old one and switching a symlink.
- Keep a local build manifest with checksums for deployed artifacts.

### Minimum readiness checks (blocking)
- PostGIS extension available.
- Disk free space above a defined threshold.
- DB migrations up to date.
- Upload bucket or local storage writable.
- GPU driver + runtime validated, plus a basic smoke test.

## Server replacement and access continuity plan
Goal: keep user access valid and trustworthy if the hosting server is replaced.

### Identity, trust, and client behavior
- Use a stable DNS name (e.g., api.mesa.example.org) and keep it constant across server changes.
- Pin a server identity via a long-lived signing key (root key) stored offline.
- Publish a rotating online signing key (active key) that is signed by the root key.
- The client trusts the root key fingerprint bundled with MESA, and trusts active keys only if the signature validates.
- If the server is replaced, ship a new active key signed by the root key; clients accept it without needing a new app build.

### Token design (portable, revocable)
- Use short-lived access tokens (JWT) signed by the active key.
- Use long-lived refresh tokens stored server-side (hashed) and client-side (encrypted in config.ini).
- When the server changes, refresh tokens remain valid because the user record is preserved and the new server can re-issue access tokens.
- Include server instance metadata in the JWT claims (issuer, key id, issued-at) for audit and diagnostics.

### Server replacement workflow
1. Provision new server with the same DNS name and TLS certificate (automated via ACME).
2. Restore the user database (tokens, quotas, audit logs) from encrypted backups.
3. Install the active key (signed by root key) and publish key metadata endpoint.
4. Bring up API endpoints, then health-checks and admin console.
5. Client tokens will refresh normally; no manual re-registration unless the user was revoked.

### Key rotation and emergency recovery
- Rotate the active key on a regular cadence (e.g., every 90 days).
- If the active key is compromised, revoke it and publish a new active key signed by the root key.
- If the root key is compromised, issue a new root key and update client trust by shipping a new MESA release.

### Client fallback and UX
- If key validation fails, show a clear message: "Server identity changed. Please retry or contact support."
- Provide a manual "Re-verify server" action in Settings to fetch and validate key metadata.
- Cache the last known good active key so transient DNS/TLS issues do not lock out users.

## Credential + server info file (admin-issued)
Goal: allow the admin to export a single file that the client can import to configure the server address, capacity, and trust details.

### File contents (JSON, signed)
- Server identity: DNS name, IP (optional), environment label.
- Capacity hints: max concurrent jobs, max upload size, GPU count/type.
- Trust bundle: root key fingerprint, active key id, key metadata URL.
- Policy hints: retention days, max job duration, supported pipeline scope.

### Delivery and import flow
1. Admin exports the file from the console.
2. File is signed by the root key (offline) or by an admin signing key that is itself signed by the root key.
3. User imports the file in MESA Settings ("Import server profile").
4. Client validates signature, stores the profile, and shows a summary for confirmation.
5. Client uses the profile as the default endpoint for registration and job submission.

### Validation and safety rules
- Reject unsigned or malformed files.
- Require DNS name to match the TLS cert when connecting.
- Show a warning if the IP is present but does not resolve to the DNS name.
- Allow multiple profiles and let the user switch between them.

### Lifecycle and versioning (concept)
- Each file has a profile id, created date, and a schema version.
- New profiles can supersede old ones but should not delete them on import.
- Client keeps the last known good profile for rollback if a new profile fails.

### How this fits the user story
- Admin sends the profile file with the distribution or by secure email.
- User imports it once, then registration and processing use the profile by default.
- If the server is replaced, admin sends a new profile with the new endpoint.
- Users do not need to re-install the app; they just import the new profile.

## Observability and reliability
- Structured logs per job (JSON lines).
- Metrics: job duration, queue wait, GPU utilization, errors.
- Alerting for job failure spikes or disk pressure.
- Retry policy for transient failures (storage or network).
- Notify the user by email when progress is measurable and when an ETA can be estimated.
- Send a completion email with links or instructions to download results.
- ETA can be derived from historical step timings, current queue depth, and rolling averages of similar job sizes.

## Multi-server awareness (seeded by admin)
Goal: allow multiple servers to discover each other and share basic metadata for routing and redundancy.

- Admin can seed each server with a shared "cluster seed" (list of peer URLs + public keys).
- Servers periodically exchange lightweight status (capacity, queue length, health, versions).
- A server can offer a read-only "cluster view" endpoint for admin diagnostics.
- Job routing remains simple: client uses its configured server unless admin enables routing.
- Seed data is updated by the admin and signed to prevent spoofing.

## Integration in MESA
- Add a Settings section: server URL + token + mode toggle.
- Add a "Process in cloud" option in the processing workflow.
- Add a cloud dialog for registration, token entry, and upload initiation.
- Local UI should show job status, progress, and last log lines.
- Outputs land in the same local output folder as local processing.

## Rollout plan
1. Prototype API and worker on a single server.
2. Implement MESA client workflow (upload -> status -> download).
3. Add admin console and security hardening.
4. Extend scope to projects and lines later.

## Open questions
- Which file formats are the minimal required assets and settings beyond GeoParquet?
- Are there any steps in the pipeline that depend on local file paths?
- How do we package and version the processing runtime on the server?
- What is the target for max job time and max upload size?
- Should we support multiple workers later or keep strict serial order?
- What is the contract for the server-side styling API and its OpenAPI docs?

## Prompt seed ideas (for later use)
- "Design a minimal REST API for token-based job submission and file transfer."
- "Define a job schema and queue policy for sequential processing with priority."
- "Draft the MESA client flow for upload, polling, and download."
- "Specify admin console endpoints for token and queue management."
