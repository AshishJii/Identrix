FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy my prebuilt dlib wheel
COPY dlib-20.0.0-cp313-cp313-linux_x86_64.whl .
RUN pip install ./dlib-20.0.0-cp313-cp313-linux_x86_64.whl

# Copy and install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files last
COPY . .

# Expose server port
EXPOSE 10000

# Run the Flask server
# CMD ["python", "server.py"]

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "server:app"]
