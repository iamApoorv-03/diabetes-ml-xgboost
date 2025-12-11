# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY diab_app/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
uvicorn diab_app.api.main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run diab_app/streamlit/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/start.sh

# Make the script executable
RUN chmod +x /app/start.sh

# Run the script on container launch
CMD ["/app/start.sh"]
