from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

if assets:
    print(assets[0])
    print(type(assets[0]))
    #for k in assets[0].keys():  #RemoteBlobAsset' object has no attribute 'keys'
    #    print(k)
else:
    print("No assets found.")