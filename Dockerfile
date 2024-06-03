# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# create folder inside code folder
RUN mkdir -p /code/storage/bm25
RUN mkdir -p /code/storage/kg

#expose the port
EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "7860"]
