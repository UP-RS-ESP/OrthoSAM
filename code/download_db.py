import requests

# Your Dropbox shared link
dropbox_url = "https://www.dropbox.com/scl/fi/f0cdqhbaxhim8ww3q4e30/Orthomosaic_hq_0.5mm_clipped.tif?rlkey=1figyrgdnhseddorzr8s0qvdl&st=ww76vha6&dl=0"

# Modify the URL to enforce direct download
direct_url = dropbox_url.replace("dl=0", "dl=1")

# Download the file
response = requests.get(direct_url, stream=True)
output_filename = "data/Argentina/Orthomosaic_hq_0.5mm_clipped.tif"

# Save the file
with open(output_filename, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"Download complete: {output_filename}")