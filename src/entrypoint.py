from service import app
from endpoints import *

# could register others here as easily once they are added to endpoints.py

if __name__ == '__main__':
    app.run(debug = True)