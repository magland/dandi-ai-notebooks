from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("001333")
assets = list(dandiset.get_assets())

# Print the paths of the first 5 assets
for i, asset in enumerate(assets[:5]):
    print(f"Asset {i+1}: {asset.path}")