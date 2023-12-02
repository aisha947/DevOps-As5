# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install scikit-learn and joblib separately
RUN pip install --trusted-host pypi.python.org scikit-learn==1.3.2 joblib>=1.1.1

# Continue with installing other dependencies
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 5000 for Flask app
EXPOSE 5000

# Copy the trained models into the container at /app
COPY model_max.joblib /app
COPY model_min.joblib /app

# Run app.py when the container launches
CMD ["python", "app.py"]
