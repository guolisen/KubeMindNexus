-- KubeMindNexus Database Initialization Script
-- This script creates all necessary tables for the KubeMindNexus application

-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    ip VARCHAR(255) NOT NULL,
    port INTEGER NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create mcp_servers table
CREATE TABLE IF NOT EXISTS mcp_servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,  -- 'stdio' or 'sse'
    command VARCHAR(255),
    args JSON,
    env JSON,
    url VARCHAR(255),
    is_local BOOLEAN DEFAULT 0,
    cluster_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Create chat_history table
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER,
    user_message TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    llm_provider VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chat_metadata JSON,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Create health_checks table
CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,  -- 'healthy', 'warning', 'error'
    details JSON,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    metric_type VARCHAR(50) NOT NULL,  -- 'cpu', 'memory', 'network', etc.
    value FLOAT NOT NULL,
    unit VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_metadata JSON,
    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE
);

-- Create config table
CREATE TABLE IF NOT EXISTS config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT,
    description TEXT
);

-- Insert default configuration values
INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('default_llm_provider', 'ollama', 'Default LLM provider');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('api_host', '127.0.0.1', 'API host address');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('api_port', '8000', 'API port');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('ui_host', '127.0.0.1', 'UI host address');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('ui_port', '8501', 'UI port');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('log_level', 'INFO', 'Logging level');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('max_mcp_retries', '3', 'Maximum MCP connection retries');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('mcp_retry_delay', '1.0', 'Delay between MCP connection retries');

INSERT OR REPLACE INTO config (key, value, description) 
VALUES ('react_max_iterations', '5', 'Maximum ReAct iterations');

-- Sample data (uncomment to use)

-- Insert sample cluster
-- INSERT INTO clusters (name, ip, port, description)
-- VALUES ('local-cluster', '127.0.0.1', 8000, 'Local Kubernetes cluster for development');

-- Insert sample MCP server
-- INSERT INTO mcp_servers (name, type, url, is_local, cluster_id)
-- VALUES ('k8s-mcp', 'sse', 'http://127.0.0.1:8000/sse', 0, 1);

-- Insert sample local MCP server
-- INSERT INTO mcp_servers (name, type, command, args, env, is_local)
-- VALUES (
--     'local-server', 
--     'stdio', 
--     'python', 
--     '["./mcp_server.py"]', 
--     '{"ENV_VAR": "value"}',
--     1
-- );
