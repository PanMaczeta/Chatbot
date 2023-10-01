FROM python:alpine
COPY . /app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD python app.py