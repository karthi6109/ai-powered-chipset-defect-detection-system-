import requests, io
from PIL import Image, ImageDraw

# create a simple test image
img = Image.new('RGB',(224,224),(255,255,255))
d = ImageDraw.Draw(img)
d.ellipse((80,60,140,120), fill=(255,0,0))
buf = io.BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
files = {'image': ('test.png', buf, 'image/png')}
try:
    r = requests.post('http://127.0.0.1:5000/analyze', files=files, timeout=20)
    print('STATUS', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('Non-json response:', r.text[:400])
except Exception as e:
    print('Request error:', e)
