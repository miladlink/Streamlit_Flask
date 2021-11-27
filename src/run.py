import os
import time
import multiprocessing


def run_flask_server():
    """ run the service """
    os.system ('./service.sh')


def run_streamlit_server():
    """ run the ui after service """
    time.sleep(7)
    os.system ('./ui.sh')


if __name__ == '__main__':

    """ Run Backend Server & Frontend Server at the same time. """

    mp1 = multiprocessing.Process(target=run_flask_server)
    mp2 = multiprocessing.Process(target=run_streamlit_server)

    mp1.start()
    mp2.start()

    mp1.join()
    mp2.join()
