import http.server
import socketserver
import threading
import time
import os

import http.server
import socketserver
import threading
import os

import http.server
import socketserver
import threading
import os

class MapServer:
    def __init__(self, port=8650, html_filename="weekly_fire_map.html"):
        self.port = port
        self.html_filename = html_filename
        self._server = None

    def start(self):
        # Create a local reference of the target file to avoid threading scope errors
        target_file = self.html_filename
        
        class FixedMapHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                # Intercept root requests or direct hits on the target HTML map file
                if self.path == '/' or self.path == f'/{target_file}':
                    if not os.path.exists(target_file):
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"Map compilation in progress. Please refresh.")
                        return
                    
                    # Calculate the explicit exact byte payload size
                    file_size = os.path.getsize(target_file)
                    
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.send_header("Content-Length", str(file_size))  # Tells the browser exactly when to stop waiting
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")  # Prevents caching stale perimeters
                    self.end_headers()
                    
                    # Write the binary stream to the socket connection
                    with open(target_file, "rb") as f:
                        self.wfile.write(f.read())
                else:
                    # Let the native handler resolve MIME types safely for standard sub-assets
                    super().do_GET()
                    
            # def log_message(self, format, *args):
            #     """Silences standard HTTP request spam in your terminal."""
            #     pass

        # Free the port socket immediately upon manual script termination (prevents address-in-use blocks)
        socketserver.TCPServer.allow_reuse_address = True
        
        try:
            self._server = socketserver.TCPServer(("", self.port), FixedMapHandler)
            
            # Spin up the server loop on a background daemon thread
            server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            server_thread.start()
            
            print(f"\n[+] Time-Slider Web App streaming live at: http://localhost:{self.port}/")
            print("[+] Control manifest data successfully parsed for machine learning pipeline pipelines.")
        except Exception as e:
            print(f"[-] Failed to launch map server on port {self.port}: {e}")

    def stop(self):
        if self._server:
            print("\nTearing down local network server socket...")
            self._server.shutdown()
            self._server.server_close()