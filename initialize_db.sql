-- KubeMindNexus Database Schema Initialization
-- This script creates the necessary tables for the KubeMindNexus application.

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Clusters table
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    ip TEXT NOT NULL,
    port INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- MCP Servers table
CREATE TABLE IF NOT EXISTS mcp_servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,  -- 'stdio' or 'sse'
    command TEXT,        -- For stdio servers
    args TEXT,           -- JSON string of command arguments
    url TEXT,            -- For SSE servers
    cluster_id INTEGER,
    is_local BOOLEAN NOT NULL DEFAULT 0,
    is_default BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- MCP Server Environment Variables
CREATE TABLE IF NOT EXISTS mcp_server_env (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    server_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    FOREIGN KEY (server_id) REFERENCES mcp_servers(id) ON DELETE CASCADE,
    UNIQUE (server_id, key)
);

-- Chat History
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    cluster_id INTEGER,
    metadata TEXT,  -- JSON string for storing additional metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Tool Executions
CREATE TABLE IF NOT EXISTS tool_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    server_name TEXT NOT NULL,
    arguments TEXT NOT NULL,  -- JSON string of arguments
    result TEXT,              -- JSON string of result
    execution_time REAL NOT NULL,  -- In seconds
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_id) REFERENCES chat_history(id) ON DELETE CASCADE
);

-- LLM Configurations
CREATE TABLE IF NOT EXISTS llm_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    parameters TEXT,  -- JSON string of parameters
    is_default BOOLEAN NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for better performance
CREATE INDEX IF NOT EXISTS idx_clusters_name ON clusters(name);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_cluster_id ON mcp_servers(cluster_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_cluster_id ON chat_history(cluster_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_chat_id ON tool_executions(chat_id);
CREATE INDEX IF NOT EXISTS idx_llm_config_is_default ON llm_config(is_default);

-- Insert default LLM configuration if none exists
INSERT OR IGNORE INTO llm_config (provider, model, is_default)
SELECT 'openai', 'gpt-4', 1 WHERE NOT EXISTS (SELECT 1 FROM llm_config);
