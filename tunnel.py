"""
Ngrok Tunnel — Expose the Streamlit app to the internet.

Usage:
    1. First, set your authtoken (one-time):
       python tunnel.py --set-token YOUR_NGROK_AUTHTOKEN

    2. Then run the tunnel:
       python tunnel.py

    Get your free authtoken at: https://dashboard.ngrok.com/signup
"""

import argparse
import sys
import time

from pyngrok import ngrok, conf


def main():
    parser = argparse.ArgumentParser(description="Expose Streamlit via ngrok tunnel")
    parser.add_argument("--port", type=int, default=8501, help="Local port (default: 8501)")
    parser.add_argument("--set-token", type=str, help="Set ngrok authtoken (one-time setup)")
    args = parser.parse_args()

    # Set authtoken if provided
    if args.set_token:
        ngrok.set_auth_token(args.set_token)
        print(f"✅ Authtoken saved! You can now run: python tunnel.py")
        return

    # Open tunnel
    print(f"🚀 Opening ngrok tunnel to localhost:{args.port}...")
    try:
        tunnel = ngrok.connect(args.port, "http")
        public_url = tunnel.public_url
        print()
        print("=" * 60)
        print(f"  🌐 PUBLIC URL:  {public_url}")
        print(f"  📱 Share this URL with any device on any network!")
        print(f"  🔗 Local:       http://localhost:{args.port}")
        print("=" * 60)
        print()
        print("Press Ctrl+C to stop the tunnel.\n")

        # Keep alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down tunnel...")
        ngrok.disconnect(tunnel.public_url)
        ngrok.kill()
        print("✅ Tunnel closed.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 If you haven't set your authtoken yet, run:")
        print(f"   python tunnel.py --set-token YOUR_TOKEN")
        print("\n   Get a free token at: https://dashboard.ngrok.com/signup")
        sys.exit(1)


if __name__ == "__main__":
    main()
