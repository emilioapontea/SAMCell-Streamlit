import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import io
import base64

def interactive_image_with_zoom(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    html = f"""
    <div id="image-container" style="position: relative;">
        <img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">
        <canvas id="highlight-canvas" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
    </div>
    <div id="zoom-container" style="margin-top: 20px;">
        <canvas id="zoom-canvas" style="border: 1px solid #ddd;"></canvas>
    </div>
    <script>
        const img = document.querySelector('#image-container img');
        const canvas = document.getElementById('highlight-canvas');
        const ctx = canvas.getContext('2d');
        const zoomCanvas = document.getElementById('zoom-canvas');
        const zoomCtx = zoomCanvas.getContext('2d');
        
        let imageData;
        const zoomFactor = 3;
        const zoomSize = 200;
        
        zoomCanvas.width = zoomSize;
        zoomCanvas.height = zoomSize;
        
        let lastZoomX, lastZoomY;  // Store the last zoom position
        
        img.onload = function() {{
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0, img.width, img.height);
            imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }};
        
        function getPixelColor(x, y) {{
            const index = (y * canvas.width + x) * 4;
            return {{
                r: imageData.data[index],
                g: imageData.data[index + 1],
                b: imageData.data[index + 2],
                a: imageData.data[index + 3]
            }};
        }}
        
        function highlightSegment(x, y) {{
            const targetColor = getPixelColor(x, y);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            for (let i = 0; i < imageData.data.length; i += 4) {{
                const r = imageData.data[i];
                const g = imageData.data[i + 1];
                const b = imageData.data[i + 2];
                
                if (r === targetColor.r && g === targetColor.g && b === targetColor.b) {{
                    const x = (i / 4) % canvas.width;
                    const y = Math.floor((i / 4) / canvas.width);
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
                    ctx.fillRect(x, y, 1, 1);
                }}
            }}
            
            // Redraw the red box if it exists
            if (lastZoomX !== undefined && lastZoomY !== undefined) {{
                drawZoomBox(lastZoomX, lastZoomY);
            }}
        }}
        
        function drawZoomBox(x, y) {{
            const scale = img.width / img.naturalWidth;
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x * scale, y * scale, (zoomSize / zoomFactor) * scale, (zoomSize / zoomFactor) * scale);
        }}
        
        function zoomImage(x, y) {{
            const scale = img.width / img.naturalWidth;
            const naturalX = x / scale;
            const naturalY = y / scale;
            
            lastZoomX = Math.max(0, Math.min(naturalX - zoomSize / (2 * zoomFactor), img.naturalWidth - zoomSize / zoomFactor));
            lastZoomY = Math.max(0, Math.min(naturalY - zoomSize / (2 * zoomFactor), img.naturalHeight - zoomSize / zoomFactor));
            
            zoomCtx.clearRect(0, 0, zoomSize, zoomSize);
            zoomCtx.drawImage(img, lastZoomX, lastZoomY, zoomSize / zoomFactor, zoomSize / zoomFactor, 0, 0, zoomSize, zoomSize);
            
            highlightSegment(x, y);  // This will also redraw the zoom box
        }}
        
        img.addEventListener('mousemove', function(event) {{
            const rect = img.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            highlightSegment(x, y);
        }});
        
        img.addEventListener('mouseleave', function() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            // Redraw the red box if it exists
            if (lastZoomX !== undefined && lastZoomY !== undefined) {{
                drawZoomBox(lastZoomX, lastZoomY);
            }}
        }});
        
        img.addEventListener('click', function(event) {{
            const rect = img.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            zoomImage(x, y);
        }});
    </script>
    """
    
    components.html(html, height=image.height + 250)

# Assuming img2 is your PIL Image object
if 'img2' in locals():
    interactive_image_with_zoom(img2, zoom_factor=3)  # Adjust zoom_factor as needed