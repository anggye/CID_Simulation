# Information spread in a social network

## About
A web-based interface to simulate the CID model on user-uploaded graphs.

## Installation

### Prerequisites
1. Clone the repository `git clone https://github.com/sawmill811/BTP.git`
2. Create a virtual environment `python -m venv venv`
3. Activate the virtual environment `venv\Scripts\activate`
4. Install all dependencies `pip install -r requirements.txt`

### Install Redis
1. Go to [this link](https://redis.io/docs/install/install-redis/install-redis-on-windows/) and follow all steps to install Redis on your system.
2. Open WSL and run `sudo service redis-server start` to start the Redis server.

### Setting up the database
1. Run `python` in the terminal to open the Python shell.
2. Run the command `from main import app` to import the Flask app.
3. Run the command `from main import db` to import the database.
4. Run the command `db.create_all()` to create the database.

## Usage

Note: for each of the following steps, you will need to open a new terminal window and make sure that the virtual environment is activated for each terminal window. 

1. In a terminal, run `celery -A main.celery  worker --loglevel=info --pool=solo` to start the Celery worker.
2. In another terminal, run `python main.py` to start the Flask server.
3. Open a web browser and go to `http://localhost:5000/` to use the application.
4. Navigate to the web page and use the application as required.
5. To close the application, press `Ctrl+C` in the terminal window where the Flask server is running.
6. Close the terminal window where the Celery worker is running.

Note : This project will be presented on 8th International Conference on Data Science and Management of Data 
https://cods-comad.in/


