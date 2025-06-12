# 💠 NamDamasco: AI-Powered Sales & Support Assistant 💠

**Version: 1.0.3** 
**Last Updated:** May 28, 2025 

## 📖 Overview

NamDamasco is an advanced Python Flask web application backend designed to serve as the intelligent core for a multi-channel conversational AI sales and support assistant. It seamlessly integrates with a customer interaction platform (like Nulu AI / Support Board), enabling businesses to offer sophisticated, AI-driven conversations on popular messaging channels like WhatsApp and Instagram (via Facebook Messenger).

The system's primary function is to understand customer inquiries in natural language, search a locally synchronized and enhanced product catalog, provide accurate product information (including details, availability, and pricing), and facilitate a smooth shopping experience. It leverages Large Language Models (LLMs) for natural language understanding and response generation, vector embeddings for semantic product search, and a robust data pipeline for keeping product information up-to-date.

## ⚙️ How It Works (Quick Summary)

1. External *Fetcher Service* sends product data to `/api/receive-products`.
2. The API validates the payload and enqueues Celery tasks.
3. Each task enriches data, generates embeddings, and upserts the database.
4. Users chat via WhatsApp/Instagram; LLM tools search products and reply.
5. Human agent replies pause the bot for that conversation.


## ✨ Core Strategy & System Architecture

NamDamasco's architecture is built around providing a highly responsive, accurate, and context-aware conversational experience. This is achieved through several key components and processes:

### 1. Data Ingestion & Asynchronous Processing Pipeline

The system relies on an external **Fetcher Service** to acquire product data from the primary Damasco inventory API. This ensures that the main NamDamasco application remains decoupled from the complexities of external API interactions and potential VPN requirements.

*   **Fetcher Service (External Component):**
    *   Periodically connects to the Damasco company's internal inventory API.
    *   Retrieves the complete product catalog, including item codes, names, **raw HTML descriptions**, categories, brands, stock levels per warehouse/branch (`almacen`/`whsName`), and prices.
    *   Securely transmits this data (typically as a list of product dictionaries with camelCase keys) to the NamDamasco application via a dedicated, authenticated API endpoint: `/api/receive-products`.

*   **NamDamasco API Endpoint (`/api/receive-products`):**
    *   **Authentication:** Validates an `X-API-KEY` from the Fetcher Service.
    *   **Basic Payload Validation:** Ensures the incoming data is a list of dictionaries.
    *   **Data Transformation:** Converts the received camelCase product data keys to snake\_case, which is the internal convention for Celery task arguments.
    *   **Asynchronous Task Enqueuing:** For each valid product item, it enqueues a background task (`process_product_item_task`) using Celery. This allows the API to respond almost instantly (HTTP 202 Accepted) to the Fetcher, acknowledging receipt and offloading the intensive processing.

*   **Celery Background Task (`process_product_item_task`):**
    This is where the core data enrichment and database operations occur for each product:
    1.  **Data Validation:** The snake\_case product data is validated using a Pydantic model (`DamascoProductDataSnake`).
    2.  **Key Case Conversion:** Data is converted back to camelCase (`product_data_camel`) for consistent interaction with internal services and model methods that expect this format for original Damasco field names.
    3.  **Conditional LLM Summarization:**
        *   The task determines if a new LLM-generated summary is needed for the product's HTML description. This occurs if the product is new, the HTML description has changed, or a summary is missing.
        *   If a new summary is required and an HTML description is available:
            *   The raw HTML is passed to `llm_processing_service.generate_llm_product_summary()`.
            *   This service first strips all HTML tags using `BeautifulSoup` (via `text_utils.strip_html_to_text`) to get plain text.
            *   The plain text is then sent to the configured LLM provider (OpenAI or Google Gemini, based on `.env` settings) with a specialized prompt to generate a concise, factual, plain-text summary (typically 50-75 words).
        *   If a new summary is not needed, the existing summary from the database is re-used.
    4.  **Text Preparation for Embedding (`Product.prepare_text_for_embedding()`):**
        *   This crucial step constructs the `searchable_text_content`.
        *   It **prioritizes the `llm_generated_summary`**. If a summary is available, it's used as the primary descriptive component.
        *   If no LLM summary is available, it falls back to using the plain text obtained by stripping the raw HTML description.
        *   This processed description is then concatenated with other key product attributes (brand, name, category, etc.) to form the final `text_to_embed`.
    5.  **Vector Embedding Generation (`openai_service.generate_product_embedding()`):**
        *   The `text_to_embed` is converted into a high-dimensional numerical vector (embedding).
    6.  **Database Upsert with Delta Detection (`product_service.add_or_update_product_in_db()`):**
        *   Efficiently updates PostgreSQL, comparing new processed values against existing records.
        *   **If no significant changes are detected**, the database write operation is skipped.
        *   If changes are found, or if the item is new, relevant fields including `description` (raw HTML), `llm_summarized_description`, `searchable_text_content`, and `embedding_vector` are stored/updated.
        *   The original `damasco_product_data` is stored in `source_data_json` for auditing.

### 2. Semantic Product Search via Vector Embeddings

*   When a user makes a product-related query, NamDamasco converts this query into a vector embedding.
*   It then performs a cosine similarity search using `pgvector` against the product embeddings in the database.
*   This allows for finding products based on meaning and context, not just keyword matches.

### 3. Intelligent LLM Interaction & Tool Usage for Conversational AI

*   User messages from the support platform are routed to a configured LLM (Google Gemini or OpenAI GPT).
*   The LLM uses "tools" (functions) to retrieve information:
    *   **`search_local_products`**: For product discovery via vector search. Returns product details including an LLM-friendly description.
    *   **`get_live_product_details`**: Retrieves detailed, up-to-date information for a specific product from the local database.

### 4. Platform Integration (e.g., Nulu AI / Support Board) for Multi-Channel Communication

*   **Incoming Messages:** A webhook endpoint (`/api/sb-webhook`) receives `message-sent` events from users on connected channels (e.g., WhatsApp, Instagram).
*   **Contextual Enrichment:** NamDamasco can use the platform's API (e.g., Support Board API) to fetch conversation history and user details, providing context to the LLM.
*   **Outgoing Replies:**
    *   **WhatsApp:** Bot replies are sent directly via the Meta WhatsApp Cloud API.
    *   **Instagram/Facebook Messenger:** Bot replies are sent through the platform's API (e.g., Support Board's `messenger-send-message`).
    *   **Dashboard Synchronization:** For all bot replies sent externally, a copy is also logged internally within the platform's conversation using its `send-message` API function. This ensures human agents have full visibility.

### 5. Differentiating Actors & Human Agent Takeover Logic (Multi-Bot Scenario)

A key aspect is distinguishing messages from different sources to ensure correct bot behavior, especially when an external "Comment Bot" initiates DMs that are ingested by the support platform.

*   **User ID Configuration in `.env` and `config.py`:**
    *   `SUPPORT_BOARD_DM_BOT_USER_ID`: The unique User ID of this NamDamasco application's bot within the support platform (e.g., User "2"). This is the ID used when NamDamasco logs its own replies.
    *   `COMMENT_BOT_PROXY_USER_ID`: The support platform User ID that is associated with the Instagram/Facebook Page itself (e.g., User "1"). When an external Comment Bot sends a DM *as the Page*, the support platform logs this message as coming from this User ID.
    *   `SUPPORT_BOARD_AGENT_IDS`: A comma-separated list of unique User IDs for *actual human agents* who use the support platform. These IDs must be distinct from the DM Bot ID and the Comment Bot Proxy ID.
    *   `COMMENT_BOT_INITIATION_TAG` (Optional): A unique string that the external Comment Bot can embed in its initial DMs. If used, this provides a more definitive way to identify messages truly initiated by the Comment Bot, even if they come from the `COMMENT_BOT_PROXY_USER_ID`.

*   **Webhook Processing Logic (`/api/sb-webhook` in `api/routes.py`):**
    1.  **DM Bot Echo:** Messages sent by `SUPPORT_BOARD_DM_BOT_USER_ID` are ignored (these are echoes of NamDamasco's own replies being logged internally).
    2.  **Comment Bot Initiated DM:**
        *   If a message's `sender_user_id` matches `COMMENT_BOT_PROXY_USER_ID` (e.g., User "1"):
            *   If `COMMENT_BOT_INITIATION_TAG` is configured and present in the message, it's confirmed as the Comment Bot.
            *   If `COMMENT_BOT_INITIATION_TAG` is *not* configured, messages from `COMMENT_BOT_PROXY_USER_ID` are *assumed* to be from the Comment Bot.
            *   In either of these "Comment Bot identified" cases, the NamDamasco DM Bot does **not** reply to this specific message and does **not** set a human takeover pause. This allows NamDamasco to respond if the *customer* replies next.
    3.  **Human Agent Intervention (Dedicated Account):**
        *   If a message's `sender_user_id` matches an ID in the `SUPPORT_BOARD_AGENT_IDS` set, it's identified as a dedicated human agent.
        *   The NamDamasco DM Bot is **paused** for this conversation for a configurable duration (set by `HUMAN_TAKEOVER_PAUSE_MINUTES`). This pause is recorded in the `conversation_pauses` database table.
    4.  **Human Agent Intervention (Proxy Account - e.g., Admin using User "1"):**
        *   If `COMMENT_BOT_INITIATION_TAG` *is* configured, and a message arrives from `COMMENT_BOT_PROXY_USER_ID` but *without* the tag, this is treated as a human (likely an admin) using that proxy account. The DM Bot is paused.
    5.  **Customer Message:**
        *   If the message is from the customer:
            *   The system first checks the `conversation_pauses` table for an explicit, active pause. If found, the DM Bot does not reply.
            *   If not explicitly paused, it then checks the recent conversation history for *implicit* human takeover. This means looking for the last message not sent by the customer, the DM Bot, or an identified Comment Bot message (using proxy ID and tag if applicable). If such a message is from any other agent ID (a dedicated human agent, or the proxy ID used by a human without the tag), the DM Bot will not reply.
            *   If no explicit pause and no implicit human takeover is detected, the NamDamasco DM Bot proceeds to process the customer's message using the configured LLM.
    6.  **Other Senders:** Any other unclassified sender is treated as potential human intervention, and the bot is paused for that conversation as a safety measure.

## 🚀 Key Features

*   📡 **Platform Webhook Integration:** Robustly handles `message-sent` events (e.g., from Nulu AI / Support Board).
*   📦 **Secure Product Data Receiver:** Authenticated `/api/receive-products` endpoint for ingesting inventory data.
*   ✨ **Asynchronous & Efficient Product Processing:** Celery-based background processing with delta detection for product updates.
*   📝 **Advanced Description Handling:** Stores raw HTML, performs conditional LLM-powered summarization, prioritizes summaries for embeddings.
*   📱 **Direct WhatsApp Cloud API Integration.**
*   🗣️ **Platform API Integration** for context and replies (e.g., Instagram/Facebook via Support Board).
*   🔎 **Intelligent Semantic Product Search** using `pgvector`.
*   🤖 **Advanced LLM Function Calling** (OpenAI/Google) with tools.
*   🐘 **PostgreSQL + `pgvector` Backend.**
*   🔄 **Decoupled Data Synchronization** via an external Fetcher Service.
*   🤝 **Differentiated Bot & Human Actor Handling:** Sophisticated logic in the webhook receiver to distinguish between the DM Bot, an external Comment Bot (via a proxy User ID), and actual Human Agents, ensuring appropriate bot behavior.
*   ⏸️ **Nuanced Human Agent Takeover Pause:** Pauses bot activity upon intervention from recognized human agents (either dedicated accounts or the proxy account if used by a human without specific bot tags) and respects these pauses for customer follow-ups.
*   ⚙️ **Environment-Based Configuration** via `.env` files for all critical settings.
*   📝 **Structured & Multi-Destination Logging.**
*   🌍 **Production-Ready Design** for Gunicorn/Caddy/Nginx.

## 📁 Project Structure
```
.
├── CHANGELOG.md
├── README.md
├── migrations/
│   └── 2024Q2_canonicalise_whs_names.sql
├── namwoo_app/
│   ├── .env.example
│   ├── .gitignore
│   ├── README.md
│   ├── __init__.py
│   ├── celery_app.py
│   ├── celery_tasks.py
│   ├── requirements.txt
│   ├── run.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── receiver_routes.py
│   │   └── routes.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── data/
│   │   ├── schema.sql
│   │   ├── store_locations.json
│   │   ├── store_locations.py
│   │   └── system_prompt.txt
│   ├── logs/
│   │   ├── app.log
│   │   └── sync.log
│   ├── models/
│   │   ├── __init__.py
│   │   ├── conversation_pause.py
│   │   └── product.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── tasks.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── damasco_service.py
│   │   ├── google_service.py
│   │   ├── llm_processing_service.py
│   │   ├── openai_service.py
│   │   ├── product_service.py
│   │   ├── support_board_service.py
│   │   └── sync_service.py
│   └── utils/
│       ├── __init__.py
│       ├── conversation_location.py
│       ├── db_utils.py
│       ├── embedding_utils.py
│       ├── product_utils.py
│       ├── string_utils.py
│       ├── text_utils.py
│       └── whs_utils.py
├── tests/
│   ├── test_end_to_end_checkout.py
│   ├── test_product_utils.py
│   ├── test_products.py
│   ├── test_prompt_flow.py
│   └── test_whatsapp_template.py
```
*(The optional `fetcher_scripts/` directory for acquiring Damasco data is maintained as a separate component and pushes updates to this application.)*

## 🛠️ Setup & Installation Guide (NamDamasco Application Server)

### Prerequisites:

*   🐍 **Python:** 3.9+
*   🐘 **PostgreSQL Server:** Version 13-16 recommended.
    *   **`pgvector` Extension:** Must be installed and enabled in your PostgreSQL database.
*   💾 **Redis Server:** Recommended for Celery message broker and result backend.
*   🐳 **Docker & Docker Compose:** Highly recommended for easily managing PostgreSQL (with `pgvector`) and Redis services.
*   🐙 **Git:** For version control.
*   🔑 **API Keys & Credentials:**
    *   Meta Developer App credentials for WhatsApp Cloud API.
    *   Support Platform (e.g., Nulu AI) installation/account with API token.
    *   LLM Provider API Key (OpenAI API Key and/or Google AI API Key for Gemini).
    *   `DAMASCO_API_SECRET`: A secret key to authenticate requests from your Fetcher Service.
*   📡 **External Fetcher Service:** Must be set up to fetch raw HTML product descriptions and send them to NamDamasco's `/api/receive-products` endpoint.
*   🤖 **External Comment Bot (Optional but relevant for full setup):** If using, ensure it sends DMs via the Instagram Page. No direct code changes needed in the Comment Bot itself if relying on proxy ID, unless implementing `COMMENT_BOT_INITIATION_TAG`.

### Installation Steps:

1.  **Clone the Repository:**
    ```bash
    git clone <your-namdamasco-repo-url>
    cd namdamasco 
    ```

2.  **Create and Activate Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `beautifulsoup4` is in `requirements.txt`)*

4.  **Set Up PostgreSQL & Redis (Docker Example):**
    *   **a. Run PostgreSQL Container (with `pgvector`):** (As before)
        ```bash
        docker run --name namwoo-postgres \
          -e POSTGRES_USER=namwoo \
          -e POSTGRES_PASSWORD=damasco2025! \
          -e POSTGRES_DB=namwoo \
          -p 5432:5432 \
          -v namwoo_postgres_data:/var/lib/postgresql/data \
          -d pgvector/pgvector:pg16 
        ```
    *   **b. Apply Database Schema & Enable `pgvector`:**
        *   Ensure your `data/schema.sql` (or migrations) defines the `products` table and the new `conversation_pauses` table.
        *   **`conversation_pauses` table schema:**
            *   `conversation_id VARCHAR(255) PRIMARY KEY`
            *   `paused_until TIMESTAMP WITH TIME ZONE NOT NULL`
        *   Execution example (if using `schema.sql`):
            ```bash
            docker cp ./data/schema.sql namwoo-postgres:/tmp/schema.sql
            docker exec -u postgres namwoo-postgres psql -d namwoo -c "CREATE EXTENSION IF NOT EXISTS vector;"
            # This next line applies your full schema, including products and conversation_pauses
            docker exec -u postgres namwoo-postgres psql -d namwoo -f /tmp/schema.sql 
            docker exec -u postgres namwoo-postgres psql -d namwoo -c "GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO namwoo;"
            docker exec -u postgres namwoo-postgres psql -d namwoo -c "GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO namwoo;"
            ```
    *   **c. Run Redis Container (for Celery):** (As before)
        ```bash
        docker run --name namwoo-redis -p 6379:6379 -v namwoo_redis_data:/data -d redis:latest redis-server --save 60 1 --loglevel warning
        ```

5.  **Configure Environment Variables (`.env` file):**
    *   Copy `cp .env.example .env`.
    *   Edit `.env` and fill in all required variables:
        *   Flask settings (`SECRET_KEY`, `FLASK_ENV`).
        *   `DATABASE_URL`.
        *   Celery settings (using lowercase keys like `broker_url`).
        *   `LLM_PROVIDER` and relevant API keys.
        *   `SUPPORT_BOARD_API_URL`, `SUPPORT_BOARD_API_TOKEN`.
        *   **`SUPPORT_BOARD_DM_BOT_USER_ID`**: The User ID of *this* NamDamasco bot in your support platform (e.g., `"2"`).
        *   **`COMMENT_BOT_PROXY_USER_ID`**: The support platform User ID that Instagram Page DMs (and thus external Comment Bot DMs) are attributed to (e.g., `"1"`).
        *   **`SUPPORT_BOARD_AGENT_IDS`**: Comma-separated string of *actual human agent* User IDs from your support platform (e.g., `"3,4,15"`). **Crucially, do not include the DM Bot ID or Comment Bot Proxy ID here.**
        *   **`COMMENT_BOT_INITIATION_TAG`**: (Optional) A unique string your Comment Bot might embed in its DMs. Leave empty if not used.
        *   `HUMAN_TAKEOVER_PAUSE_MINUTES` (e.g., `43200` for 30 days).
        *   WhatsApp Cloud API credentials.
        *   `DAMASCO_API_SECRET`.

6.  **Database Migrations (If using Flask-Migrate/Alembic):**
    *   If managing schema with Alembic, create and apply migrations for the `conversation_pauses` table and any changes to the `products` table.

7.  **Run Initial Data Sync (Trigger External Fetcher):**
    *   Ensure your external Fetcher Service is configured.
    *   Execute it to populate NamDamasco with initial product data.

8.  **Run NamDamasco Application (Development/Testing):**
    *   **Terminal 1: Flask Application Server (Gunicorn)**
        ```bash
        gunicorn --bind 0.0.0.0:5100 "run:app" --log-level debug --worker-class gevent --workers 4 --timeout 300
        ```
    *   **Terminal 2: Celery Worker(s)**
        ```bash
        celery -A namwoo_app.celery_app worker -l INFO -P gevent -c 2 
        ```

9.  **Configure Support Platform Webhook (e.g., Nulu AI):**
    *   **URL:** `https://your-public-domain-or-ngrok-url.com/api/sb-webhook`
    *   Ensure the `message-sent` event (or equivalent) is active.

10. **Test Thoroughly:**
    *   **Data Ingestion & Processing:** (As before)
    *   **Conversational AI & Tool Use:** (As before)
    *   **Actor Differentiation & Pause Logic:**
        *   **Scenario: Comment Bot Initiates -> User Replies -> DM Bot Replies:**
            1.  User comments on IG.
            2.  External Comment Bot sends initial DM (seen in Support Board as from `COMMENT_BOT_PROXY_USER_ID`).
            3.  *Verify: NamDamasco logs this, DM Bot does not reply to this message, no pause is set.*
            4.  User replies to this DM.
            5.  *Verify: NamDamasco DM Bot identifies customer reply, sees no active pause, determines no overriding human intervention (recognizing Comment Bot's message is not human), and replies.*
        *   **Scenario: Human Agent (Dedicated Account) Intervenes:**
            1.  After any bot interaction, a human agent (with User ID from `SUPPORT_BOARD_AGENT_IDS`) replies via the support platform.
            2.  *Verify: NamDamasco DM Bot receives this webhook, identifies it as a dedicated human agent, and sets a pause for the conversation in the `conversation_pauses` table.*
            3.  User replies again.
            4.  *Verify: NamDamasco DM Bot sees the active pause and does not reply.*
        *   **Scenario: Human Admin (Using Proxy ID "1") Intervenes:**
            1.  If `COMMENT_BOT_INITIATION_TAG` is configured:
                *   Admin (as User "1") sends a DM *without* the tag. *Verify: NamDamasco treats as human intervention and pauses.*
            2.  If `COMMENT_BOT_INITIATION_TAG` is *not* configured:
                *   Admin (as User "1") sends a DM. *Verify (based on current logic): NamDamasco may treat this as the Comment Bot and *not* pause. This highlights the importance of the tag or strict rules for User "1".*
        *   **Scenario: Direct DM from User -> DM Bot -> Human Agent -> User -> No DM Bot Reply:** (As before)
    *   **Database Verification:** Check `products` (for descriptions, summaries, embeddings) and `conversation_pauses` tables.

11. **Production Deployment:**
    *(As before: systemd, reverse proxy, cron for fetcher)*

## 💡 Important Considerations & Future Enhancements

*   **Error Handling & Resilience.**
*   **API Rate Limits.**
*   **Security:**
    *   Protect API keys and credentials.
    *   Emphasize distinct User IDs for human agents. If the `COMMENT_BOT_PROXY_USER_ID` (e.g., User "1") *must* also be used by a human admin for DMs, implementing the `COMMENT_BOT_INITIATION_TAG` in the Comment Bot's DMs is highly recommended for accurate differentiation by NamDamasco.
*   **Scalability.**
*   **Vector Database Optimization.**
*   **Advanced Location-Based Search/Filtering.**
*   **Cost Management.**
*   **Idempotency.**
