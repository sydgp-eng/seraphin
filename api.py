from fastapi import FastAPI
from pydantic import BaseModel
from rag_chain import get_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SERA Trust Georgia DV Assistant",
    description="AI assistant that provides general information about Georgia domestic violence laws and resources."
)

# Allow embedding on websites (seratrust.org)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict this later to only your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = get_chain()

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(data: Query):
    answer = agent(data.question)
    return {"answer": answer}
