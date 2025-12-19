import requests
import json

# Configuration
URL = "http://localhost:8000/analyze_video"
VIDEO_PATH = "path/to/your/poultry_video.mp4" # Update this to your file name

def test_poultry_api():
    try:
        # Open the file in binary mode
        with open(VIDEO_PATH, "rb") as f:
            # Define the multipart form data
            files = {"file": (VIDEO_PATH, f, "video/mp4")}
            params = {
                "fps_sample": 5,
                "conf_thresh": 0.25
            }

            print(f"🚀 Sending {VIDEO_PATH} to API...")
            response = requests.post(URL, files=files, params=params)

        # Check if request was successful
        if response.status_code == 200:
            data = response.json()
            print("✅ Success!")
            print(f"📊 Average Count: {sum(x['count'] for x in data['counts']) / len(data['counts']):.2f}")
            print(f"📂 Video saved at: {data['artifacts']['video_url']}")
            
            # Save the JSON response to a file for inspection
            with open("api_results.json", "w") as out:
                json.dump(data, out, indent=4)
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except FileNotFoundError:
        print("❌ Error: Video file not found. Please check VIDEO_PATH.")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Server not running. Run 'python main.py' first.")

if __name__ == "__main__":
    test_poultry_api()