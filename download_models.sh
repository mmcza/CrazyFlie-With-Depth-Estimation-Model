#!/bin/bash

# Path to the file containing the model links
MODEL_LINKS_FILE="models_links.txt"

# Check if the file exists
if [[ ! -f "$MODEL_LINKS_FILE" ]]; then
    echo "File $MODEL_LINKS_FILE does not exist."
    exit 1
fi

# Read the file line by line
while IFS= read -r line; do
    # Remove quotes if present and trim whitespace
    line=$(echo "$line" | tr -d '"' | xargs)
    
    # Execute the gz fuel download command
    gz fuel download -v 4 -u "$line"
done < "$MODEL_LINKS_FILE"
