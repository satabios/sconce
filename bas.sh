#!/bin/bash

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Get the git root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

# Function to convert bytes to human readable format
format_size() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes}B"
    elif [ $bytes -lt $((1024 * 1024)) ]; then
        echo "$((bytes / 1024))KB"
    elif [ $bytes -lt $((1024 * 1024 * 1024)) ]; then
        echo "$((bytes / 1024 / 1024))MB"
    else
        echo "$((bytes / 1024 / 1024 / 1024))GB"
    fi
}

echo "Current directory: $(pwd)"
echo "Git root directory: $GIT_ROOT"
echo "------------------------"
echo "Calculating folder sizes..."
echo "------------------------"

# List all directories first to check if we have any
dirs=$(find . -type d -not -path "*/\.*" 2>/dev/null)
if [ -z "$dirs" ]; then
    echo "No directories found in current path."
    exit 1
fi

# Use du to calculate folder sizes, sort by size (largest first)
du -sh */ .* 2>/dev/null | \
sort -hr | \
while read -r size path; do
    if [ -d "$path" ] && [ "$path" != "." ] && [ "$path" != ".." ]; then
        printf "%-40s %s\n" "${path%/}" "$size"
    fi
done

# If no output was produced, show all files instead
if [ $? -ne 0 ]; then
    echo "No directories found. Listing files instead:"
    ls -lah | grep -v '^total'
fi

echo "------------------------"