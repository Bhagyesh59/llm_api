import os
from fastapi import FastAPI
from app import endpoints
from app.models import Summary
from routers.items import ChatCompletionRequest
from app.agents import summary
import uvicorn

root_path = "/"
if os.getenv("ROUTE"):
    root_path = os.getenv("ROUTE")


app = FastAPI(root_path=root_path)

app.include_router(endpoints.router)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

# @app.get("/health")
# async def healthCheck():
#     return {"message": "success"}

# @app.post("/summary", response_model=Summary)
# async def summary_agent(request: ChatCompletionRequest):
#     data=request.data
#     return summary.summary_responce(data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
    