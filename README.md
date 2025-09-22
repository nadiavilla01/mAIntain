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








now you must ask yourself where are the previous commit...since i always committed to local and when it came to push everything to the repo i made for the project it was impossible, kept figuring out for 2 hours end even when i fixed the problem the issue continue to pop up.
in the end i decided to eliminte all the commits becasue they were giving me issues and push the only commit it remained aka the last changes.
I hope this won't get penalized becasue its a pity and honestly since everything in the istructions was so strict , it could've been useful to have more guidance on git, and how git repo would have been evaluated.
i am going to attach proof of the git error 




<img width="1440" height="900" alt="Screenshot 2025-09-22 at 00 41 15" src="https://github.com/user-attachments/assets/86099201-52c3-4398-8a41-12774e0e31bb" />





<img width="1440" height="900" alt="Screenshot 2025-09-22 at 00 41 16" src="https://github.com/user-attachments/assets/48301d93-ae57-4a40-a188-99862dbd54ec" />

it was jsut 2 hours like this :)
