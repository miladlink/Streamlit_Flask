from flask import Flask
from service_warmup import init

app = Flask (__name__)
app = init (app)