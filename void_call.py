from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (including file:// as 'null')
    allow_credentials=True,
    allow_methods=["*"],          # Allow POST, OPTIONS, etc.
    allow_headers=["*"],          # Allow Content-Type, Authorization, etc.
)

model = SentenceTransformer("all-mpnet-base-v2")

@app.get("/")
def read_root():
    return FileResponse("void.html")

@app.post("/void")
async def void(request: Request):
    entries = open("void.txt").readlines()
    entries = [entry.strip() for entry in entries]
    embeddings = np.loadtxt("void.csv", delimiter=",", dtype=np.float32)
    print(f"Loaded {len(entries)} entries and {embeddings.shape[0]} embeddings.")

    data = await request.json()
    entry = data["message"]
    
    if not entry:
        return JSONResponse(content={"response": "silence"})
    elif entry in entries:
        return JSONResponse(content={"response": "silence"})
    else:
        # encode the input
        encoding = model.encode(entry)
        # compare with the existing entries
        similarity = model.similarity(encoding, embeddings)
        best_idx = np.argmax(similarity[0])
        response = entries[best_idx]
        
        entries.append(entry)
        embeddings = np.vstack([embeddings, encoding])
        open("void.txt", "w").write("\n".join(entries))
        np.savetxt("void.csv", embeddings.astype(np.float32), delimiter=",")
        
        print(response)
        return JSONResponse(content={"response": response})

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run(app)