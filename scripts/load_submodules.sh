#!/bin/bash

# Initialize submodules if not already initialized
git submodule init

# Update all submodules to their latest committed state
git submodule update

# Initialize nested submodules if any exist
git submodule foreach --recursive 'git submodule init'
git submodule foreach --recursive 'git submodule update'

echo "All submodules have been loaded successfully"
