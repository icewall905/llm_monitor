#!/bin/bash
# switch-llm.sh — Dynamic LLM stack switcher
# Discovers model stacks from stacks/*.yml via LLM_META header comments.
# Usage:
#   ./switch-llm.sh                        # interactive menu
#   ./switch-llm.sh --stack stacks/X.yml   # switch directly (used by dashboard)
#   ./switch-llm.sh --stop-all             # stop all stacks

# LLAMA_DIR must be set by the environment (passed via Docker container)
if [ -z "$LLAMA_DIR" ]; then
    echo "Error: LLAMA_DIR environment variable is not set."
    exit 1
fi

STACKS_DIR="$LLAMA_DIR/stacks"
GPU_MONITOR_DIR="/mnt/nvmestorage/compose/llama.cpp/gpu-monitor"

# ---------------------------------------------------------------------------
# Metadata extraction — reads "# LLM_META field="value"" lines from a file
# ---------------------------------------------------------------------------
get_meta() {
    local file="$1" field="$2"
    grep "^# LLM_META ${field}=" "$file" 2>/dev/null \
        | head -1 \
        | sed "s/^# LLM_META ${field}=\"//;s/\"$//"
}

# ---------------------------------------------------------------------------
# Discover stacks — emit sorted list of file paths (by sort_order)
# ---------------------------------------------------------------------------
discover_stacks() {
    for f in "$STACKS_DIR"/*.yml; do
        [ -f "$f" ] || continue
        order="$(get_meta "$f" "sort_order")"
        order="${order:-999}"
        # Accept integer/decimal sort orders (e.g., 2.5). Fallback to 999.
        if ! [[ "$order" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            order="999"
        fi
        printf '%s\t%s\n' "$order" "$f"
    done | sort -t $'\t' -k1,1g -k2,2 | cut -f2-
}

# ---------------------------------------------------------------------------
# Stop all discovered stacks
# ---------------------------------------------------------------------------
stop_all() {
    echo "Stopping any running containers from all stacks..."
    cd "$LLAMA_DIR" || exit 1
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        docker compose --project-directory "$LLAMA_DIR" -p llamacpp -f "$f" down --remove-orphans 2>/dev/null
    done < <(discover_stacks)
    # Force-stop any remaining containers with the llama working-dir label
    local leftover
    leftover=$(docker ps -q \
        --filter "label=com.docker.compose.project.working_dir=${LLAMA_DIR}" 2>/dev/null)
    if [ -n "$leftover" ]; then
        echo "Force-stopping lingering containers in ${LLAMA_DIR}:"
        echo "$leftover" | xargs docker stop 2>/dev/null || true
        echo "$leftover" | xargs docker rm -f 2>/dev/null || true
    fi
    echo "All stacks stopped."
}

# ---------------------------------------------------------------------------
# Start a specific stack (by relative or absolute path)
# ---------------------------------------------------------------------------
start_stack() {
    local file="$1"
    local display_name
    display_name="$(get_meta "$file" "display_name")"
    display_name="${display_name:-$file}"

    echo "Switching to: $display_name"
    stop_all

    cd "$LLAMA_DIR" || exit 1
    if [ -f "$file" ]; then
        if ! docker compose --project-directory "$LLAMA_DIR" -p llamacpp -f "$file" up -d; then
            echo "Warning: docker compose reported a startup failure for $display_name."
            echo "Check logs with: docker compose -p llamacpp logs -f"
            return 1
        fi
        echo "$display_name started successfully."
        echo "Active containers:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

        # Start gpu-monitor if not already running
        if ! docker ps --format '{{.Names}}' | grep -q '^gpu-monitor$'; then
            docker compose -f "$GPU_MONITOR_DIR/docker-compose.yml" \
                --project-name gpu-monitor up -d
        fi
    else
        echo "Error: Compose file not found: $file"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# CLI flag handling
# ---------------------------------------------------------------------------
case "${1:-}" in
    --stop-all)
        stop_all
        exit 0
        ;;
    --stack)
        if [ -z "${2:-}" ]; then
            echo "Usage: $0 --stack stacks/model.yml"
            exit 1
        fi
        # Accept relative (to LLAMA_DIR) or absolute path
        target="$2"
        if [[ "$target" != /* ]]; then
            target="$LLAMA_DIR/$target"
        fi
        start_stack "$target"
        exit 0
        ;;
    "")
        # Interactive menu — fall through
        ;;
    *)
        echo "Unknown flag: $1"
        echo "Usage: $0 [--stack stacks/model.yml] [--stop-all]"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------
mapfile -t STACK_FILES < <(discover_stacks)

if [ ${#STACK_FILES[@]} -eq 0 ]; then
    echo "No stacks found in $STACKS_DIR"
    exit 1
fi

echo ""
echo "Select the LLM stack to deploy:"
i=1
for f in "${STACK_FILES[@]}"; do
    display_name="$(get_meta "$f" "display_name")"
    display_name="${display_name:-$(basename "$f" .yml)}"
    printf '%d) %s\n' "$i" "$display_name"
    i=$((i + 1))
done
stop_num=$i
echo "${stop_num}) Stop All"
echo "q) Quit"
echo ""

read -rp "Enter choice [1-${stop_num}]: " choice

if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
    echo "Exiting."
    exit 0
fi

if [[ "$choice" == "$stop_num" ]]; then
    stop_all
    exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#STACK_FILES[@]}" ]; then
    echo "Invalid choice: $choice"
    exit 1
fi

selected_file="${STACK_FILES[$((choice - 1))]}"
start_stack "$selected_file"
