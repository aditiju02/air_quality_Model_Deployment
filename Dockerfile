FROM python:3.9
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
RUN pip install --upgrade pip

# Set the working directory in the container
WORKDIR /app/

# Copy the application files into the working directory
COPY . .

RUN python -m venv /opt/env

# # Enable venv
ENV PATH="/opt/env/bin:${PATH}"

# Install the application dependencies
RUN pip install -r requirements.txt

EXPOSE 8080
CMD ["python", "./app.py"]
