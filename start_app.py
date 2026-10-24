import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


URL = "http://127.0.0.1:8000"
HEALTH_URL = f"{URL}/api/health"


def is_running() -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main() -> None:
    if is_running():
        print(f"Sentiment Studio is already running: {URL}", flush=True)
        webbrowser.open(URL)
        return

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    server = subprocess.Popen(command)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"Web server stopped during startup (exit code {server.returncode})."
                )
            if is_running():
                print(f"Sentiment Studio is ready: {URL}", flush=True)
                webbrowser.open(URL)
                break
            else:
                time.sleep(1)
        else:
            raise TimeoutError("Sentiment Studio did not start within 60 seconds.")
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        server.wait(timeout=10)
    except Exception:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=10)
        raise


if __name__ == "__main__":
    main()
