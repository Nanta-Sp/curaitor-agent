# Curaitor Agent  
**AI agent for scientific data extraction**  
Part of Schmidt OxRSE Workshop (Sep 11–20, 2025)  

---

## Overview  
Curaitor Agent (Nanta-Sp fork) consists of two parts
- DataInitializer
  -  run once to initialize the database
- Curaitor-agent
  - for automatic tracking and update database, chat with the database, send email notification

---

## Quick Start  
1) Install uv (Python package and project manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2) Clone repo and cd to the repo
3) Initialize environment
```bash
uv sync
```
4) Edit config file
  - choose the model you want to use under llm:
    - provider: openai
    - model: "gpt-5-mini"

5) Provide gmail address
  - send your gmail email address to nsophonrat2@gmail.com to be added to the user pool

6) add .env file
    Create .env file in the agent folder with your 
    ```bash
    OPENAI_API_KEY=
    OPENROUTER_API_KEY=
    GMAIL_CREDENTIALS_PATH=
    GMAIL_TOKEN_PATH=secrets/token.json
    ```
7) Initialise the database
```bash
uv run python curaitor_agent/data_initializer.py --pdf-dir papers
```
  - Replace `papers` with the folder containing your PDFs. The script respects
    `config.yaml` defaults for chunking and embeddings, and accepts overrides
    such as `--chunk-size 800 --chunk-overlap 80 --embedding-model sentence-transformers/all-MiniLM-L6-v2`.

8) Run gmail authentication
This will work for 1 hour.
```bash
uv run python curaitor_agent_v2/gmail_create_token.py
```

9) Run web interface
```bash
uv run adk web
```

### Functions you can use
#### curaitor_agent
- search and summarize paper from arxiv
- schedule time of day for daily search
- send email summary to yourself

---

## For Developer
### Dependency Management  
- Add a new package:  
  ```bash
  uv add package-name
  uv sync
  ```


---

### MCP Inspector Tool  

The **MCP Inspector** helps verify your MCP server connection and test available tools.  

1) Install requirements  
- [nvm](https://github.com/nvm-sh/nvm) (Node Version Manager)  
- Node.js ≥ 18 (v22 recommended)
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
   \. "$HOME/.nvm/nvm.sh"

   nvm install 22

   node -v   # v22.19.0
   npm -v    # 10.9.3
   ```

2) Run the MCP Inspector:  
   ```bash
   npx @modelcontextprotocol/inspector uv run tools/mcp_server.py
   ```

3) In the MCP Inspector UI, click **Connect** → test tools.

---


---

## License  
This project is licensed under the **MIT License**.  
