          # Use the stable Python 3.11 version for AI libraries
FROM python:3.11-slim

# Set the working directory inside the cloud computer
WORKDIR /app

# Copy your requirements and code into the container
COPY . .

# Install dependencies and download the Spacy NLP model
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Hugging Face Spaces require Port 7860
EXPOSE 7860

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]