Terminal 1: ML Microservice

cd ml-service
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8001 --reload

# Health check: curl http://127.0.0.1:8001/health

Terminal 2: Backend API

cd server
npm install
npm run migrate
npm run dev
# Health check: curl http://localhost:5000/api/health

Terminal 3: Frontend Client

cd client
npm install
npm run dev
# Open browser at: http://localhost:5173