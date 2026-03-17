PROVIDER = "LLM Llama 3.1 8B model"  

API_KEY = "utsa-jABQlGLaTrae2bqMHyAvPxTvE9KTP0DEWYIXhvtgkDkVcGjp44rN6G56x1aGiyem"
BASE_URL = "http://149.165.173.247:8888/v1"  
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"               

TEMPERATURE_DEBATER = 0.7
TEMPERATURE_JUDGE = 0.2
MAX_TOKENS = 500
NUM_ROUNDS = 3

DATA_PATH = "data/sample_questions.json"
LOG_PATH = "logs/debates.json"
RESULT_PATH = "results/summary.json"


DEBUG = True
PRINT_EVERY_CALL = True
SAVE_AFTER_EACH_QUESTION = True
MAX_QUESTIONS = 100   