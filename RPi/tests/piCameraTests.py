from picamera2 import Picamera2, Preview
from io import BytesIO
from time import sleep
from PIL import Image

# Create the in-memory stream
stream = BytesIO()
camera = Picamera2()
camera.start_preview(Preview.QTGL)

preview_config = camera.create_preview_configuration(main={"size": (800, 600)})
camera.start()
sleep(2)
# "Rewind" the stream to the beginning so we can read its content
camera.capture_file(stream, format = 'jpeg')
stream.seek(0)
image = Image.open(stream)