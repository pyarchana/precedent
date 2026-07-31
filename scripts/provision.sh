#!/usr/bin/env bash
#
# Provision the CockroachDB Cloud cluster for Precedent.
#
# Everything here goes through the ccloud CLI rather than the web console, so
# the cluster is reproducible from a file rather than from memory of which
# buttons were clicked. Re-running is safe: each step checks for what it is
# about to create.
#
#   bash scripts/provision.sh
#
# Prerequisites: ccloud installed (see below) and `ccloud auth login` done once.

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-precedent}"
CLOUD="${CLOUD:-AWS}"
REGION="${REGION:-ap-south-1}"     # Mumbai, nearest to the developer
SQL_USER="${SQL_USER:-precedent_app}"
DATABASE="${DATABASE:-precedent}"

# Basic tier, capped so it can never bill. The free monthly resources cover a
# demo of this size, and a hard zero means an accident cannot become a charge.
SPEND_LIMIT="${SPEND_LIMIT:-0}"

ccloud_bin() {
    if command -v ccloud >/dev/null 2>&1; then
        echo "ccloud"
    elif [ -x "$APPDATA/ccloud/ccloud.exe" ]; then
        echo "$APPDATA/ccloud/ccloud.exe"
    else
        cat >&2 <<'EOF'
ccloud is not installed.

macOS or Linux:
    brew install cockroachdb/tap/ccloud

Windows PowerShell:
    $ProgressPreference = 'SilentlyContinue'
    New-Item -Type Directory -Force $env:APPDATA/ccloud | Out-Null
    Invoke-WebRequest -Uri https://binaries.cockroachdb.com/ccloud/ccloud_windows-amd64_0.6.12.zip -OutFile $env:TEMP/ccloud.zip
    Expand-Archive -Force -Path $env:TEMP/ccloud.zip -DestinationPath $env:TEMP/ccloud_extract
    Copy-Item -Force $env:TEMP/ccloud_extract/ccloud.exe -Destination $env:APPDATA/ccloud

Then authenticate once:
    ccloud auth login
EOF
        exit 1
    fi
}

CCLOUD="$(ccloud_bin)"

echo "==> Checking authentication"
if ! "$CCLOUD" cluster list >/dev/null 2>&1; then
    echo "Not authenticated. Run: $CCLOUD auth login" >&2
    exit 1
fi

echo "==> Looking for an existing cluster named $CLUSTER_NAME"
if "$CCLOUD" cluster list --output json | grep -q "\"name\": *\"$CLUSTER_NAME\""; then
    echo "    already exists, leaving it alone"
else
    echo "==> Creating $CLUSTER_NAME on $CLOUD in $REGION, spend limit \$$SPEND_LIMIT"
    "$CCLOUD" cluster create serverless "$CLUSTER_NAME" "$REGION" \
        --cloud "$CLOUD" \
        --spend-limit "$SPEND_LIMIT" \
        --wait
fi

echo "==> Cluster details"
"$CCLOUD" cluster info "$CLUSTER_NAME"

# A dedicated SQL user rather than the account owner. The MCP server on day 16
# gets its own read-only role separately; this one is what the application uses.
echo "==> Ensuring SQL user $SQL_USER"
if "$CCLOUD" cluster user list "$CLUSTER_NAME" 2>/dev/null | grep -q "$SQL_USER"; then
    echo "    already exists, leaving it alone"
else
    "$CCLOUD" cluster user create "$CLUSTER_NAME" "$SQL_USER"
    echo "    password printed above, it is shown once only"
fi

cat <<EOF

==> Next steps

1. Put the connection string in .env as COCKROACH_DSN, inserting the password
   you chose during the user creation step above:

       $CCLOUD cluster sql "$CLUSTER_NAME" --connection-url \\
           --database "$DATABASE" --username "$SQL_USER"

2. Apply the schema:

       python -m precedent.db.migrate --create-db

3. Load the corpus, which never re-hits the GitHub API:

       python -m precedent.transform.run

Nothing here bills while the spend limit is \$$SPEND_LIMIT.
EOF
