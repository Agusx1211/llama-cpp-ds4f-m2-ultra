# Trusted subdomain gateway

This is the production Caddy edge for the single loopback-only M2 Ultra
`llama-server`. Four distinct HTTPS DNS names select `low`, `normal`, `fast`,
or admin traffic. The proxy deletes all client scheduling credentials and
classification fields before adding its private backend token and the lane
fixed by the requested hostname. The admin host injects only the operator
token, exposes `/internal/admin/*` to the backend, and serves the immutable
dashboard for every other path.

The dashboard's **Connect this HTTPS gateway** path uses only its exact page
origin, sends the admin Bearer in memory, and omits the private operator header;
Caddy installs that header after stripping browser input. The older explicit
URL/operator-token form remains available only for direct loopback operation.
For admin calls, Caddy replaces `Origin` with the exact loopback backend origin
required by the server but preserves `Sec-Fetch-Site`; a cross-site request
therefore still fails the server's same-site gate even if a key is disclosed.

Each API hostname requires its own exact, URL-safe Bearer key before proxying;
the admin dashboard itself remains static and credential-free, while every
`/internal/admin/*` request requires the admin key. Configure the same four
keys in `llama-server`, so its normal API-key check remains a second boundary.
Caddy validation rejects reused keys: possession of a low or normal key must
not authorize the fast or admin hostname.
Caddy does not log request headers and its admin API is disabled. The backend
must be started with an explicit loopback bind:

```sh
export LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN="$(cat /run/secrets/llama-trusted-token)"
llama-server --host 127.0.0.1 --port 18130 --api-key-file /run/secrets/llama-api-keys ...
```

Configure public A/AAAA records for the four names to the gateway address,
then provide the exact sites and private runtime values. `TLS_MODE` is normally
the ACME account email; `internal` is only for a closed validation network.

```sh
export M2_GATEWAY_LOW_HOST=low.llm.example.com
export M2_GATEWAY_NORMAL_HOST=normal.llm.example.com
export M2_GATEWAY_FAST_HOST=fast.llm.example.com
export M2_GATEWAY_ADMIN_HOST=admin.llm.example.com
export M2_GATEWAY_ACME_EMAIL=operator@example.com
export M2_GATEWAY_TLS_MODE=operator@example.com
export M2_GATEWAY_BACKEND=http://127.0.0.1:18130
export M2_GATEWAY_TRUSTED_TOKEN="$(cat /run/secrets/llama-trusted-token)"
export M2_GATEWAY_LOW_API_KEY="$(cat /run/secrets/llama-low-api-key)"
export M2_GATEWAY_NORMAL_API_KEY="$(cat /run/secrets/llama-normal-api-key)"
export M2_GATEWAY_FAST_API_KEY="$(cat /run/secrets/llama-fast-api-key)"
export M2_GATEWAY_ADMIN_API_KEY="$(cat /run/secrets/llama-admin-api-key)"
export M2_GATEWAY_DASHBOARD_ROOT=/opt/llama-m2/current/dashboard

python3 tools/m2-deploy/gateway/validate_gateway.py \
  --expected-address 203.0.113.10 --run
```

Validation fails on missing or duplicate hostnames, non-production site
addresses, DNS answers outside the predeclared gateway addresses, a nonliteral
or nonloopback backend, an invalid trusted token, a missing/oversized/writable
dashboard tree, a non-root-owned production canonical path or asset, any
dashboard symlink or special file, or a Caddy
adaptation/validation error. Use every active A and AAAA gateway address
as a repeated `--expected-address`; omitting the option still requires every
name to resolve.

Production must use the validator's `--run` mode rather than invoking Caddy in
a second command. The Caddyfile reads only the launcher's private pinned-root
environment value and a required-argument launch guard, so direct Caddyfile
adaptation or launch without the validator fails. The launcher
resolves the Caddy binary, Caddyfile, and release dashboard once; requires
their canonical components and assets to be root-owned and not group/world
writable; validates Caddy with those canonical paths; and then replaces itself
with the same Caddy process and environment. Retargeting an atomic `current`
release or configuration symlink after launch
therefore cannot change the files served by the running gateway; changing the
canonical release requires root authority and a gateway restart.

Run the adversarial integration test with a Caddy 2.9 or later binary. It starts
a real TLS reverse proxy and a loopback recording backend, then attempts exact,
mixed-case, duplicate, and multi-header priority forgery:

```sh
CADDY=/path/to/caddy python3 tools/m2-deploy/gateway/test_gateway.py
```

The environment carries the shared operator token because both Caddy and
`llama-server` need it. Supply it from a service-manager secret file at process
start, restrict process inspection to the deployment account, and never place
it in the Caddyfile, release manifest, command line, DNS, URL, or logs.
