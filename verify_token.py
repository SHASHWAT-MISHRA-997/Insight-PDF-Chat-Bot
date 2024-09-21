import requests

# Use your Hugging Face API token here
huggingfacehub_api_token = "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"

if not huggingfacehub_api_token:
    print("API token is not set. Please set it in the script.")
    exit()

# Test API endpoint for model list
test_url = "https://huggingface.co/api/models"

headers = {
    "Authorization": f"Bearer {huggingfacehub_api_token}"
}

try:
    response = requests.get(test_url, headers=headers)
    if response.status_code == 200:
        print("API token is valid and working.")
    else:
        print(f"Failed to verify API token. Status code: {response.status_code}")
        print("Response:", response.json())
except Exception as e:
    print(f"Error verifying API token: {e}")
