FROM python:3.10-slim

WORKDIR /src/recognition

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "pub_sub_subcriber.py"]
