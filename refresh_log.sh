#!/bin/bash

# Clear the screen initially for a clean start
clear

# Loop until the user hits Ctrl+C
while true; do
    # Display the current time so you know when it last refreshed
    echo "Last refreshed: $(date)"
    echo "------------------------------------------"
    
    # Run your command
    tail -20 sets/render_all.log
    
    # Wait for 30 seconds
    sleep 30
    
    # Clear the screen for the next refresh (optional)
    clear
done
