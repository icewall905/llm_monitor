#!/bin/bash

# HTML output path (nginx default root)
OUTPUT_FILE="/usr/share/nginx/html/index.html"

# Generate HTML page in a loop
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    SMI_OUTPUT=$(nvidia-smi 2>&1 || echo "Error: Failed to run nvidia-smi. Make sure nvidia-smi and libnvidia-ml.so.1 are mounted correctly.")

    cat > "$OUTPUT_FILE" << HTML_EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="5">
    <title>GPU Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background-color: #1a1a2e;
            color: #eee;
            font-family: 'Courier New', Courier, monospace;
            padding: 20px;
            min-height: 100vh;
        }
        h1 {
            color: #00d9ff;
            margin-bottom: 20px;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .info {
            color: #888;
            text-align: center;
            margin-bottom: 20px;
            font-size: 14px;
        }
        pre {
            background-color: #16213e;
            border: 1px solid #0f3460;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
            font-size: 14px;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU Monitor</h1>
        <p class="info">Auto-refreshing every 5 seconds | Last updated: ${TIMESTAMP}</p>
        <pre>${SMI_OUTPUT}</pre>
    </div>
</body>
</html>
HTML_EOF

    sleep 5
done
