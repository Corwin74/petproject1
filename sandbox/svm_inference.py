from joblib import load
from sklearn.preprocessing import scale

WORK_DIR = '/home/alex/projects/petproject1/sandbox/data/'

def predict(img):
    svm = load(WORK_DIR + 'svm.joblib')
    pred = svm.predict(img)
    return pred
