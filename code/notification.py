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

with open('code/DWH.txt', 'r') as file:
    tail = file.readline().strip()
head='https://discord.com/api/webhooks/'

send_discord_alert(head+tail, f"{script_name} has completed successfully!")