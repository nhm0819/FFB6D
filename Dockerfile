FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# Install linux packages
RUN apt update

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# Copy contents
COPY . /app
