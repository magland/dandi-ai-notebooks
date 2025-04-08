#!/bin/bash

# Get list of dandiset directories
for dandiset in dandisets/[0-9]*; do
    # Skip if already a submodule
    if [ -f "$dandiset/.git" ]; then
        echo "Skipping $dandiset - already a submodule"
        continue
    fi

    dandiset_id=$(basename $dandiset)
    echo "Processing $dandiset_id..."

    # Create GitHub repository
    gh repo create dandi-ai-notebooks/$dandiset_id --public

    # Initialize git and push content
    (cd $dandiset && \
        git init && \
        git add . && \
        git commit -m "Initial commit" && \
        git branch -M main && \
        git remote add origin https://github.com/dandi-ai-notebooks/$dandiset_id.git && \
        git push -u origin main)

    # Remove from main repo and add as submodule
    git rm -r $dandiset
    git submodule add https://github.com/dandi-ai-notebooks/$dandiset_id.git $dandiset

    echo "Completed $dandiset_id"
    echo "-------------------"
done

echo "All dandisets processed"
