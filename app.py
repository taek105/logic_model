from init import xor_model, or_model, not_model, and_model

from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/")
def root():
    return {"message": "connection online"}

@app.get("/and")
def or_inference(x1: int = Query(), x2: int = Query()):
    result = and_model.predict([x1, x2])
    return {"x1": x1, "x2": x2, "AND result": result}

@app.get("/and/test")
def or_test():
    return and_model.testJson()

@app.get("/or")
def or_inference(x1: int = Query(), x2: int = Query()):
    result = or_model.predict([x1, x2])
    return {"x1": x1, "x2": x2, "OR result": result}

@app.get("/or/test")
def or_test():
    return or_model.testJson()

@app.get("/not")
def not_inference(x1: int = Query()):
    result = not_model.predict(x1)
    return {"x1": x1, "NOT result": result}

@app.get("/not/test")
def not_test():
    return not_model.testJson()

@app.get("/xor")
def xor_inference(x1: int = Query(), x2: int = Query()):
    result = xor_model.predict([x1, x2])
    return {"x1": x1, "x2": x2, "XOR result": result}

@app.get("/xor/test")
def xor_test():
    return xor_model.testJson()