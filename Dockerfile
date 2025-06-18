FROM python:3.11
LABEL authors="ramtahor"

#WORKDIR /usr/src/app

COPY requirements.txt ./
COPY viewer.py ./

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

#COPY . .

CMD ["python", "./viewer.py"]