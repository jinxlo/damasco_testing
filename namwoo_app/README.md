# 💠 NamDamasco: AI-Powered Sales & Support Assistant 💠

**Version: 1.0.4** 
**Last Updated:** June 11, 2025 

## 📖 Overview

NamDamasco is an advanced Python Flask web application backend designed to serve as the intelligent core for a multi-channel conversational AI sales and support assistant. It seamlessly integrates with a customer interaction platform (like Nulu AI / Support Board), enabling businesses to offer sophisticated, AI-driven conversations on popular messaging channels like WhatsApp and Instagram (via Facebook Messenger).

The system's primary function is to understand customer inquiries in natural language, search a locally synchronized and enhanced product catalog, provide accurate product information (including details, availability, and pricing), and facilitate a smooth shopping experience. It leverages Large Language Models (LLMs) for natural language understanding and response generation, vector embeddings for semantic product search, and a robust data pipeline for keeping product information up-to-date. A key part of the sales flow involves collecting essential customer details and sending a WhatsApp template message for order confirmation, after which the conversation is routed to a human sales agent.

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
*   It then performs a cosine similarity search using `pgvector` against the product embeddings in the database. This search can be filtered by `warehouse_names` (obtained via the `get_store_info` tool based on user's city) and optionally by a price range (if specified by the user and extracted by the LLM).
*   This allows for finding products based on meaning and context, not just keyword matches, and tailored to location and budget.

### 3. Intelligent LLM Interaction & Tool Usage for Conversational AI

*   User messages from the support platform are routed to a configured LLM (Google Gemini or OpenAI GPT).
*   The LLM uses "tools" (functions) to retrieve information and perform actions:
    *   **`get_store_info`**: Retrieves store locations, addresses, and `whsName` (warehouse identifiers) based on a city name from a local JSON file. This is used to determine relevant stores for local product searches and provide store details.
    *   **`search_local_products`**: For product discovery via vector search. Can be filtered by `warehouse_names` and a price range (e.g., `min_price`, `max_price`). Returns product details including an LLM-friendly description.
    *   **`get_live_product_details`**: Retrieves detailed, up-to-date information for a specific product from the local database. Used for final stock checks before order confirmation.
    *   **`initiate_customer_information_collection`**: Called once the user confirms a product and chooses a non-Cashea payment method. Registers initial lead interest with an external Lead API.
    *   **`submit_customer_information_for_crm`**: Called after collecting essential customer details (Full Name, Cedula, Phone, Email) to update the lead information via the external Lead API.
    *   **`send_whatsapp_order_summary_template`**: After all data is collected by the LLM and stock is confirmed, this tool sends a pre-defined WhatsApp template (e.g., `confirmacion_datos_cliente`) to the user's WhatsApp number via the Support Board `messaging-platforms-send-template` API function. This template contains order details for final user confirmation.
        *   Upon successful sending of this template, the original conversation (e.g., on Instagram or Telegram) is automatically routed to the Sales department within Support Board, and bot interaction is paused for that conversation.

### 4. Platform Integration (e.g., Nulu AI / Support Board) for Multi-Channel Communication

*   **Incoming Messages:** A webhook endpoint (`/api/sb-webhook`) receives `message-sent` events from users on connected channels (e.g., WhatsApp, Instagram).
*   **Contextual Enrichment:** NamDamasco can use the platform's API (e.g., Support Board API) to fetch conversation history and user details, providing context to the LLM.
*   **Outgoing Replies & Template Messaging:**
    *   **WhatsApp (Standard Replies):** Bot replies are sent directly via the Meta WhatsApp Cloud API.
    *   **WhatsApp (Template Messages):** Order confirmation templates (e.g., `confirmacion_datos_cliente`) are sent via the Support Board `messaging-platforms-send-template` API function, using details collected by the LLM and the customer's Support Board user ID.
    *   **Instagram/Facebook Messenger:** Bot replies are sent through the platform's API (e.g., Support Board's `messenger-send-message`).
    *   **Dashboard Synchronization:** For all bot replies sent externally (including direct WhatsApp messages and template messages sent via SB), a copy is also logged internally within the platform's conversation using its `send-message` API function. This ensures human agents have full visibility.

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
            *   **(Note: If the customer replies "Sí" to a WhatsApp order confirmation template, the webhook logic in `api/routes.py` should ideally identify this specific context and route the WhatsApp conversation to the Sales department, potentially pausing further bot replies in that specific WhatsApp thread too.)**
    6.  **Other Senders:** Any other unclassified sender is treated as potential human intervention, and the bot is paused for that conversation as a safety measure.

## 🚀 Key Features

*   📡 **Platform Webhook Integration:** Robustly handles `message-sent` events (e.g., from Nulu AI / Support Board).
*   📦 **Secure Product Data Receiver:** Authenticated `/api/receive-products` endpoint for ingesting inventory data.
*   ✨ **Asynchronous & Efficient Product Processing:** Celery-based background processing with delta detection for product updates.
*   📝 **Advanced Description Handling:** Stores raw HTML, performs conditional LLM-powered summarization, prioritizes summaries for embeddings.
*   📱 **Direct WhatsApp Cloud API Integration** (for standard bot replies).
*   💬 **WhatsApp Template Messaging:** Sends pre-approved WhatsApp templates (e.g., for order confirmation) via the Support Board API, populated with data collected by the LLM.
*   🗣️ **Platform API Integration** for context and replies (e.g., Instagram/Facebook via Support Board).
*   🔎 **Intelligent Semantic Product Search** using `pgvector`, with support for filtering by location (via `whsName`) and price range.
*   🏪 **Dynamic Store Information Retrieval:** LLM uses a tool (`get_store_info`) to query a local JSON file for store details (addresses, `whsName`) based on city.
*   🤖 **Advanced LLM Function Calling** (OpenAI/Google) with tools for store lookup, product search, live product details, lead/CRM interaction, and sending WhatsApp templates.
*   📋 **Guided Sales Data Collection:** LLM-driven process to gather essential customer information (name, ID, phone, email) for order processing and template personalization.
*   ↪️ **Automated Conversation Routing to Sales:** After successfully sending a WhatsApp order confirmation template, the original conversation (e.g., Instagram) is automatically assigned to the Sales department and the bot is paused for that specific conversation.
*   🐘 **PostgreSQL + `pgvector` Backend.**
*   🔄 **Decoupled Data Synchronization** via an external Fetcher Service.
*   🤝 **Differentiated Bot & Human Actor Handling:** Sophisticated logic in the webhook receiver to distinguish between the DM Bot, an external Comment Bot (via a proxy User ID), and actual Human Agents, ensuring appropriate bot behavior.
*   ⏸️ **Nuanced Human Agent Takeover Pause:** Pauses bot activity upon intervention from recognized human agents (either dedicated accounts or the proxy account if used by a human without specific bot tags) and respects these pauses for customer follow-ups.
*   ⚙️ **Environment-Based Configuration** via `.env` files for all critical settings (including Sales Department ID).
*   📝 **Structured & Multi-Destination Logging.**
*   🌍 **Production-Ready Design** for Gunicorn/Caddy/Nginx.

## 📁 Folder Structure (NamDamasco Application Server)
/NAMDAMASCO_APP_ROOT/
|-- namwoo_app/ # Main application package
| |-- init.py # App factory (create_app), main app config
| |-- api/
| | |-- init.py # Defines 'api_bp' Blueprint, imports route modules
| | |-- receiver_routes.py # Handles /api/receive-products (enqueues Celery tasks)
| | |-- routes.py # Handles /api/sb-webhook, /api/health
| |-- celery_app.py # Celery application setup (with Flask context management)
| |-- celery_tasks.py # Celery task definitions (product processing, summarization)
| |-- config/
| | |-- config.py # Defines Config class, loads .env
| |-- data/ # Static data, e.g., LLM system prompts, store locations
| | |-- system_prompt.txt
| | |-- store_locations.json # NEW: Stores location data for the `get_store_info` tool
| |-- models/
| | |-- init.py # Defines SQLAlchemy Base, imports all models
| | |-- product.py # Product ORM model (with description, llm_summarized_description)
| | |-- conversation_pause.py # ConversationPause ORM model
| |-- services/
| | |-- init.py # Exposes service functions/modules for easy import
| | |-- damasco_service.py # Helper for initial processing of raw Damasco data (outputs snake_case)
| | |-- google_service.py # Google Gemini specific logic (chat, summarization)
| | |-- openai_service.py # OpenAI specific logic (chat, embedding, summarization, tool definitions including `get_store_info` and `send_whatsapp_order_summary_template`)
| | |-- product_service.py # Core logic for DB ops, vector search (now with price filtering), delta detection
| | |-- support_board_service.py # Platform API interactions (e.g., Nulu AI / Support Board, including `messaging-platforms-send-template` and `route_conversation_to_sales`)
| | |-- sync_service.py # Coordinates bulk data sync (can call Celery or product_service)
| | |-- llm_processing_service.py # Dispatches summarization to configured LLM provider
| |-- utils/
| | |-- init.py
| | |-- db_utils.py # Database session management, pause logic
| | |-- embedding_utils.py # Helper for calling embedding models
| | |-- text_utils.py # Contains strip_html_to_text
| | |-- product_utils.py # Shared product ID generation logic
| | |-- conversation_location.py # Manages user location context (city, relevant whsNames)
| |-- scheduler/ # APScheduler related tasks (if used for other cron jobs)
| |-- init.py
| |-- tasks.py
|-- data/ # Project-level data (e.g., SQL schema if not using migrations)
| |-- schema.sql # Must include 'description', 'llm_summarized_description', and 'conversation_pauses' table
|-- logs/ # Created at runtime for log files
|-- venv/ # Python virtual environment (.gitignored)
|-- .env # Environment variables (SECRET! .gitignored)
|-- .env.example # Example environment variables
|-- .gitignore
|-- requirements.txt # Python dependencies (add beautifulsoup4)
|-- run.py # Entry point for Gunicorn (e.g., run:app which calls create_app)
|-- gunicorn.conf.py # (Optional) Gunicorn configuration file
|-- Caddyfile # Example Caddy reverse proxy configuration
|-- README.md # This file
*(Note: The `fetcher_scripts/` directory for Damasco data acquisition is considered a separate, complementary project/component that pushes data to this application.)*

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
*   📄 **Store Locations File:** Ensure `namwoo_app/data/store_locations.json` is created and populated with your store details.

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
        *   **`SUPPORT_BOARD_SALES_DEPARTMENT_ID`**: The numeric ID of your Sales department in Support Board (e.g., "2").
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
    *   **Conversational AI & Tool Use:**
        *   Test `get_store_info` tool by asking for stores in a city.
        *   Test `search_local_products` with and without `warehouse_names`, and with queries that include price ranges.
        *   Test `get_live_product_details`.
    *   **WhatsApp Template Flow:**
        1.  User expresses interest in a product and chooses "Pago de Contado" or "Pagar en Tienda".
        2.  Bot collects Name, Cedula, Phone, Email.
        3.  Verify the `initiate_customer_information_collection` and `submit_customer_information_for_crm` tools are called in the correct order.
        4.  Verify `get_live_product_details` is called for stock check.
        5.  Verify the `send_whatsapp_order_summary_template` tool is called by the LLM.
        6.  Verify the WhatsApp template message is received by the user with correctly populated variables (including derived store address for pickup, or generic shipping message).
        7.  Verify the original conversation (e.g., Instagram) is routed to the Sales Department in Support Board and the bot is paused for that conversation.
        8.  Test the scenario where the user replies "Sí" to the WhatsApp template (this part of the routing would be handled by your webhook logic in `api/routes.py` to route the WhatsApp conversation itself to Sales).
    *   **Actor Differentiation & Pause Logic:** (As before)
    *   **Database Verification:** Check `products` (for descriptions, summaries, embeddings) and `conversation_pauses` tables.

11. **Production Deployment:**
    *(As before: systemd, reverse proxy, cron for fetcher)*

## 💡 Important Considerations & Future Enhancements

*   **WhatsApp Template Management:** Ensure the `confirmacion_datos_cliente` template is approved by Meta and correctly set up in your WhatsApp Business API / Support Board integration. Parameter order and count must match what the `send_whatsapp_order_summary_template` tool prepares. El campo "Dirección" de esa plantilla debe contener **solo** la sucursal donde se retirará el producto, nunca la dirección del cliente.
*   **Error Handling & Resilience.**
*   **API Rate Limits.**
*   **Security:**
    *   Protect API keys and credentials.
    *   Emphasize distinct User IDs for human agents. If the `COMMENT_BOT_PROXY_USER_ID` (e.g., User "1") *must* also be used by a human admin for DMs, implementing the `COMMENT_BOT_INITIATION_TAG` in the Comment Bot's DMs is highly recommended for accurate differentiation by NamDamasco.
*   **Scalability.**
*   **Vector Database Optimization.**
*   **Advanced Location-Based Search/Filtering.** (Price range filtering added, further refinements possible).
*   **Cost Management.**
*   **Idempotency.**