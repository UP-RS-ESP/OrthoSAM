import requests
import sys

script_name=sys.argv[1]
def send_discord_alert(webhook_url, message):
    data = {"content": message}
    try:
        response = requests.post(webhook_url, json=data)
        if response.status_code == 204:
            print("Message sent successfully to Discord!")
        else:
            print(f"Failed to send message: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

url='https://discord.com/api/webhooks/1322110114521415730/V_wCZ6s6PiPj48KOEybUya-amw3KZTH2iLAb411DxhLDVtez9My2DPu0SlNNK-wlszoJ'

send_discord_alert(url, f"{script_name} has completed successfully!")