To run everything in the backend:
start for each pre trained model:
run the training algorithms to generate folders and pth since git doesn't let me push the models.
them in the backend folder run: uvicorn main:app --reload
then to run the frontend run npm install, make sure you are using node 22
and do npm run dev.

how to be sure everything works correctly:
backend should return this
P Backend % uvicorn main:app --reload

INFO:     Will watch for changes in these directories: ['/Users/nadiavilla/Desktop/FINAAAAAL/Backend']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [85705] using StatReload
[BOOT] OPENAI_API_KEY present: False
[Insights] OPENAI_API_KEY not set; LLM disabled.
Loaded fault classifier from /Users/nadiavilla/Desktop/FINAAAAAL/Backend/routes/../pretrainedResNet/fault_classifier_resnet18.pth
INFO:     Started server process [85710]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

INFO:     127.0.0.1:59194 - "GET /machines?mode=start HTTP/1.1" 200 OK
INFO:     127.0.0.1:59197 - "GET /machines?mode=start HTTP/1.1" 200 OK

and frontend this
up to date, audited 419 packages in 864ms

90 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities

> ai-dashboard@0.0.0 dev
> vite


  VITE v7.1.7  ready in 199 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help




















