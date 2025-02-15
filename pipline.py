from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/pipeline")
async def receive_data(request: Request):
    try:
        data = await request.json()
        print("Received Data:", json.dumps(data, indent=4))  # Print formatted data in terminal
        return {"message": "Data received successfully"}
    except Exception as e:
        return {"error": str(e)}
