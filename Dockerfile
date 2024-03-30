# Use a Python 3.9 (or higher) base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for custom CMake arguments
ENV set "CMAKE_ARGS=-DLLAMA_OPENBLAS=on"
ENV set "FORCE_CMAKE=1"
RUN pip install llama-cpp-python --no-cache-dir


# Copy the rest of your application's source code from your host to your image filesystem.
COPY . /app

# Command to run on container start
CMD ["python", "gpu-app.py"]