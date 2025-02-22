from init import xor_model, or_model, not_model

from fastapi import FastAPI, Query

# print(f"ormodel 0,0: {or_model.predict([0,0])}")
# print(f"notmodel 1: {not_model.predict(1)}")
# print(f"xormodel 1,1: {xor_model.predict([1,1])}")

app = FastAPI()

@app.get("/or")
def or_inference(x1: int = Query(), x2: int = Query()):
    result = or_model.predict([x1, x2])
    return {"x1": x1, "x2": x2, "OR result": result}

@app.get("/or/test")
def or_test():
    return or_model.testJson()

@app.get("/not")
def not_inference(x: int = Query()):
    result = not_model.predict(x)
    return {"x": x, "NOT result": result}

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