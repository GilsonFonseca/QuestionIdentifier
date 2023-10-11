To create a module you will need

1. Dockerfile
2. module-name-specs.json
3. requirements.txt

Dockerfile example:

"
FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["src/worker.py"]
"

module-name-specs example:

"
{
  "id": "question-identifier",
  "name": "Question Identifier",
  "description": "Receives a text and returns the questions found",
  "author": "Gilson Fonseca",
  "email": "gilson@ice.ufjf.br",
  "input_queue": "question-identifier-python-in",
  "output_queue": "question-identifier-python-out",
  "input": [
    {
      "id": "input-file",
      "link": true,
      "type": "file",
      "required": true
    }
  ],
  "output": [
    {
      "id": "input-file",
      "link": true,
      "type": "file",
      "required": true
    }
  ]
}
"

Example of Requirements:
"
nltk
tensorflow
sklearn
git+https://github.com/maxjf1/files-ms-client-python#egg=files-ms-client-python //necessary
git+https://github.com/easytopic-project/client_python_m2p //necessary
"


Then just upload it to a public repository