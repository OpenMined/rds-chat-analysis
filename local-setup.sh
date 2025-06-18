#!/bin/bash
# filepath: /home/azureuser/dev/syftbox/syftbox-setup.sh

set -e

# Configuration variables
CLIENT_COUNT=2
BASE_EMAIL="test@test.com"
SERVER_BUILD_TAGS="sonic avx nomsgpack"
CLIENT_BUILD_TAGS="go_json nomsgpack"
SYFTBOX_CONFIG_DIR=~/.syftbox
DATA_ROOT="/home/azureuser/Desktop/SyftBox"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --clients) CLIENT_COUNT="$2"; shift ;;
        --base-email) BASE_EMAIL="$2"; shift ;;
        --data-root) DATA_ROOT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Build common variables
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUILD_COMMIT=$(git rev-parse --short HEAD || echo "dev")
SYFTBOX_VERSION="0.5.0"
BUILD_LD_FLAGS="-s -w -X github.com/openmined/syftbox/internal/version.Version=${SYFTBOX_VERSION} -X github.com/openmined/syftbox/internal/version.Revision=${BUILD_COMMIT} -X github.com/openmined/syftbox/internal/version.BuildDate=${BUILD_DATE}"

echo "Building SyftBox components..."

# Extract server details from .env file
FULL_ADDRESS=$(grep SYFTBOX_HTTP_ADDRESS .env | sed 's/.*"\([^"]*\)".*/\1/')
SERVER_HOST=$(echo "$FULL_ADDRESS" | cut -d ':' -f1)
SERVER_PORT=$(echo "$FULL_ADDRESS" | cut -d ':' -f2)
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

if [[ -z "$SERVER_HOST" || -z "$SERVER_PORT" ]]; then
    SERVER_URL="http://localhost:8080"
    echo "Warning: Could not extract server details from .env, using default: $SERVER_URL"
fi

# Check if tmux is installed, if not try to install it
if ! command -v tmux &> /dev/null; then
    echo "tmux not found, trying to install it..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Function to build the server
build_server() {
    echo "Building server binary..."
    go build -trimpath --tags="${SERVER_BUILD_TAGS}" \
        -ldflags="${BUILD_LD_FLAGS}" \
        -o ./syftbox_server ./cmd/server
    
    chmod +x ./syftbox_server
    echo "Server binary built successfully at ./syftbox_server"
}

# Function to build the client
build_client() {
    echo "Building client binary..."
    go build -trimpath --tags="${CLIENT_BUILD_TAGS}" \
        -ldflags="${BUILD_LD_FLAGS}" \
        -o ./syftbox_client ./cmd/client
    
    chmod +x ./syftbox_client
    echo "Client binary built successfully at ./syftbox_client"
}

# Generate email for a client
generate_client_email() {
    local client_num=$1
    local prefix=${BASE_EMAIL%@*}
    local domain=${BASE_EMAIL#*@}
    printf "%s+%02d@%s" "$prefix" "$client_num" "$domain"
}

# Kill existing tmux sessions
cleanup_tmux_sessions() {
    echo "Cleaning up existing tmux sessions..."
    
    # Kill server session if it exists
    if tmux has-session -t syftbox_server 2>/dev/null; then
        echo "Killing existing syftbox_server session..."
        tmux kill-session -t syftbox_server
    fi
    
    # Kill client sessions if they exist
    for ((i=1; i<=CLIENT_COUNT; i++)); do
        if tmux has-session -t "syftbox_client_${i}" 2>/dev/null; then
            echo "Killing existing syftbox_client_${i} session..."
            tmux kill-session -t "syftbox_client_${i}"
        fi
    done
    
    # Give tmux time to fully clean up
    sleep 1
}

# Build the binaries
build_server
build_client

# Clean up existing sessions
cleanup_tmux_sessions

# Create required directories
mkdir -p "${DATA_ROOT}/server"
mkdir -p ${SYFTBOX_CONFIG_DIR}

# Start tmux session for server with explicit session creation
echo "Starting server at ${SERVER_URL}..."
tmux new-session -d -s syftbox_server
tmux send-keys -t syftbox_server "./syftbox_server" C-m
echo "Server started in tmux session 'syftbox_server'"

# Wait for server to start
sleep 3

# Start clients
for ((i=1; i<=CLIENT_COUNT; i++)); do
    CLIENT_EMAIL=$(generate_client_email $i)
    CONFIG_FILE="${SYFTBOX_CONFIG_DIR}/config_client_${i}.json"
    
    
    # Start the client in a new tmux session with explicit commands
    echo "Starting client ${i} with email ${CLIENT_EMAIL}..."
    tmux new-session -d -s "syftbox_client_${i}"
    tmux send-keys -t "syftbox_client_${i}" "cd $(pwd)" C-m
    tmux send-keys -t "syftbox_client_${i}" "./syftbox_client login -c ${CONFIG_FILE} -s http://localhost:8080 " C-m
    echo "Client ${i} started in tmux session 'syftbox_client_${i}'"
    
    # Wait a moment between starting clients
    sleep 1
done

echo "SyftBox deployment complete!"
echo "- Server running at ${SERVER_URL}"
echo "- ${CLIENT_COUNT} clients running with base email ${BASE_EMAIL}"
echo "- Client configs located in ${SYFTBOX_CONFIG_DIR}"
echo "- Data directories in ${DATA_ROOT}"
echo ""
echo "To view server logs: tmux attach -t syftbox_server"
echo "To view client logs: tmux attach -t syftbox_client_<number>"
echo "To detach from a tmux session: press Ctrl+B, then D"
echo "To list all tmux sessions: tmux ls"