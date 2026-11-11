
# Use Python 3.11 as the base environment
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing cache files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python output immediately
ENV PYTHONUNBUFFERED=1

# Copy our project into the container
COPY . .

# Install our project and all its dependencies
RUN pip install --no-cache-dir -e .

# Streamlit uses port 8501
EXPOSE 8501

# Start the Streamlit application
CMD ["python", "-m", "streamlit", "run", "src/epl_predictor/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
