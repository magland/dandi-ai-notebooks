#!/bin/bash

# Initialize submodules if not already initialized
git submodule init

# Update all submodules recursively (pulls latest commits)
git submodule update --recursive --remote

# Enter each submodule and ensure we're on the main branch with latest changes
git submodule foreach 'git checkout main || git checkout master && git pull'
