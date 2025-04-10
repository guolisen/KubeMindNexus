"""Database management for KubeMindNexus."""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from .config import Configuration

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for KubeMindNexus."""
    
    def __init__(self, config: Configuration):
        """Initialize database manager.
        
        Args:
            config: Configuration instance.
        """
        self.config = config
        self.db_path = self.config.get_db_path()
        self.conn = None
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database connection and tables."""
        try:
            # Create parent directory if needed
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # Initialize schema
            self._init_schema()
            
            logger.info(f"Connected to database at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        try:
            # Get script file path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(script_dir, "..", "initialize_db.sql")
            
            if not os.path.exists(schema_path):
                # If not found in parent directory, try current directory
                schema_path = os.path.join(script_dir, "initialize_db.sql")
                
            if not os.path.exists(schema_path):
                # If still not found, raise error
                raise FileNotFoundError(f"Schema file not found at {schema_path}")
                
            # Read schema script
            with open(schema_path, "r") as f:
                schema_script = f.read()
                
            # Execute schema script
            self.conn.executescript(schema_script)
            self.conn.commit()
            
            logger.info("Database schema initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database schema: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    # Cluster management
    def add_cluster(
        self, name: str, ip: str, port: int, description: Optional[str] = None
    ) -> int:
        """Add a new cluster.
        
        Args:
            name: Cluster name.
            ip: Cluster IP address.
            port: Cluster port.
            description: Optional cluster description.
            
        Returns:
            Cluster ID.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO clusters (name, ip, port, description)
                VALUES (?, ?, ?, ?)
                """,
                (name, ip, port, description),
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding cluster: {str(e)}")
            raise
    
    def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get a cluster by ID.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            Cluster data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM clusters WHERE id = ?
                """,
                (cluster_id,),
            )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting cluster: {str(e)}")
            raise
    
    def get_cluster_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a cluster by name.
        
        Args:
            name: Cluster name.
            
        Returns:
            Cluster data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM clusters WHERE name = ?
                """,
                (name,),
            )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting cluster by name: {str(e)}")
            raise
    
    def get_all_clusters(self) -> List[Dict[str, Any]]:
        """Get all clusters.
        
        Returns:
            List of cluster data.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM clusters")
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting all clusters: {str(e)}")
            raise
    
    def update_cluster(
        self,
        cluster_id: int,
        name: Optional[str] = None,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a cluster.
        
        Args:
            cluster_id: Cluster ID.
            name: New cluster name.
            ip: New cluster IP address.
            port: New cluster port.
            description: New cluster description.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            # Get current cluster data
            current = self.get_cluster(cluster_id)
            if not current:
                return False
                
            # Update fields
            updates = {}
            if name is not None:
                updates["name"] = name
            if ip is not None:
                updates["ip"] = ip
            if port is not None:
                updates["port"] = port
            if description is not None:
                updates["description"] = description
                
            if not updates:
                # Nothing to update
                return True
                
            # Build update query
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            query = f"UPDATE clusters SET {set_clause} WHERE id = ?"
            
            # Execute update
            cursor = self.conn.cursor()
            cursor.execute(
                query,
                list(updates.values()) + [cluster_id],
            )
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating cluster: {str(e)}")
            return False
    
    def delete_cluster(self, cluster_id: int) -> bool:
        """Delete a cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            True if the deletion was successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                DELETE FROM clusters WHERE id = ?
                """,
                (cluster_id,),
            )
            self.conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting cluster: {str(e)}")
            return False
    
    # MCP server management
    def add_mcp_server(
        self,
        name: str,
        type_str: str,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        cluster_id: Optional[int] = None,
        is_local: bool = False,
        is_default: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> int:
        """Add a new MCP server.
        
        Args:
            name: Server name.
            type_str: Server type ('stdio' or 'sse').
            command: Command to execute (for stdio servers).
            args: Command arguments (for stdio servers).
            url: Server URL (for sse servers).
            cluster_id: Optional cluster ID to associate with.
            is_local: Whether this is a local server.
            is_default: Whether this is a default server.
            env: Environment variables for the server.
            
        Returns:
            Server ID.
        """
        try:
            # Serialize command arguments and environment variables as JSON
            args_json = json.dumps(args) if args else None
            env_json = json.dumps(env) if env else None
            
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO mcp_servers (
                    name, type, command, args, url, cluster_id, is_local, is_default
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name, type_str, command, args_json, url, cluster_id,
                    1 if is_local else 0, 1 if is_default else 0,
                ),
            )
            server_id = cursor.lastrowid
            
            # Add environment variables if provided
            if env and server_id:
                for key, value in env.items():
                    cursor.execute(
                        """
                        INSERT INTO mcp_server_env (server_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        (server_id, key, value),
                    )
            
            self.conn.commit()
            
            # Update configuration if this is a default server
            if is_default:
                self.config.add_default_mcp_server(name)
                
            return server_id
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding MCP server: {str(e)}")
            raise
    
    def get_mcp_server(self, server_id: int) -> Optional[Dict[str, Any]]:
        """Get an MCP server by ID.
        
        Args:
            server_id: Server ID.
            
        Returns:
            Server data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM mcp_servers WHERE id = ?
                """,
                (server_id,),
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            server = dict(row)
            
            # Deserialize args from JSON
            if server["args"]:
                try:
                    server["args"] = json.loads(server["args"])
                except json.JSONDecodeError:
                    server["args"] = []
            else:
                server["args"] = []
                
            # Get environment variables
            cursor.execute(
                """
                SELECT key, value FROM mcp_server_env
                WHERE server_id = ?
                """,
                (server_id,),
            )
            
            env = {row["key"]: row["value"] for row in cursor.fetchall()}
            server["env"] = env
            
            return server
            
        except Exception as e:
            logger.error(f"Error getting MCP server: {str(e)}")
            raise
    
    def get_mcp_server_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get an MCP server by name.
        
        Args:
            name: Server name.
            
        Returns:
            Server data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id FROM mcp_servers WHERE name = ?
                """,
                (name,),
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            return self.get_mcp_server(row["id"])
            
        except Exception as e:
            logger.error(f"Error getting MCP server by name: {str(e)}")
            raise
    
    def get_all_mcp_servers(self) -> List[Dict[str, Any]]:
        """Get all MCP servers.
        
        Returns:
            List of server data.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM mcp_servers")
            
            servers = []
            for row in cursor.fetchall():
                server = self.get_mcp_server(row["id"])
                if server:
                    servers.append(server)
                    
            return servers
            
        except Exception as e:
            logger.error(f"Error getting all MCP servers: {str(e)}")
            raise
    
    def get_mcp_servers_by_cluster(self, cluster_id: int) -> List[Dict[str, Any]]:
        """Get MCP servers for a specific cluster.
        
        Args:
            cluster_id: Cluster ID.
            
        Returns:
            List of server data.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id FROM mcp_servers WHERE cluster_id = ?
                """,
                (cluster_id,),
            )
            
            servers = []
            for row in cursor.fetchall():
                server = self.get_mcp_server(row["id"])
                if server:
                    servers.append(server)
                    
            return servers
            
        except Exception as e:
            logger.error(f"Error getting MCP servers by cluster: {str(e)}")
            raise
    
    def update_mcp_server(
        self,
        server_id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        cluster_id: Optional[int] = None,
        is_local: Optional[bool] = None,
        is_default: Optional[bool] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Update an MCP server.
        
        Args:
            server_id: Server ID.
            name: New server name.
            type: New server type.
            command: New command.
            args: New command arguments.
            url: New server URL.
            cluster_id: New cluster ID.
            is_local: New local flag.
            is_default: New default flag.
            env: New environment variables.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            # Get current server data
            current = self.get_mcp_server(server_id)
            if not current:
                return False
                
            # Update fields
            updates = {}
            if name is not None:
                updates["name"] = name
            if type is not None:
                updates["type"] = type
            if command is not None:
                updates["command"] = command
            if args is not None:
                updates["args"] = json.dumps(args)
            if url is not None:
                updates["url"] = url
            if cluster_id is not None:
                updates["cluster_id"] = cluster_id
            if is_local is not None:
                updates["is_local"] = 1 if is_local else 0
            if is_default is not None:
                updates["is_default"] = 1 if is_default else 0
                
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
                
            # Update server record if needed
            if updates:
                # Build update query
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                query = f"UPDATE mcp_servers SET {set_clause} WHERE id = ?"
                
                # Execute update
                cursor = self.conn.cursor()
                cursor.execute(
                    query,
                    list(updates.values()) + [server_id],
                )
            
            # Update environment variables if provided
            if env is not None:
                cursor = self.conn.cursor()
                
                # Delete existing environment variables
                cursor.execute(
                    """
                    DELETE FROM mcp_server_env WHERE server_id = ?
                    """,
                    (server_id,),
                )
                
                # Add new environment variables
                for key, value in env.items():
                    cursor.execute(
                        """
                        INSERT INTO mcp_server_env (server_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        (server_id, key, value),
                    )
            
            # Commit transaction
            self.conn.commit()
            
            # Update configuration if default flag changed
            if is_default is not None:
                current_name = current["name"]
                new_name = name or current_name
                
                if is_default and new_name not in self.config.get_default_mcp_servers():
                    self.config.add_default_mcp_server(new_name)
                elif not is_default and new_name in self.config.get_default_mcp_servers():
                    self.config.remove_default_mcp_server(new_name)
                
                # If name changed and was in default servers, update name
                if name and name != current_name and current_name in self.config.get_default_mcp_servers():
                    self.config.remove_default_mcp_server(current_name)
                    self.config.add_default_mcp_server(name)
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating MCP server: {str(e)}")
            return False
    
    def delete_mcp_server(self, server_id: int) -> bool:
        """Delete an MCP server.
        
        Args:
            server_id: Server ID.
            
        Returns:
            True if the deletion was successful, False otherwise.
        """
        try:
            # Get server data for configuration update
            server = self.get_mcp_server(server_id)
            if not server:
                return False
                
            cursor = self.conn.cursor()
            cursor.execute(
                """
                DELETE FROM mcp_servers WHERE id = ?
                """,
                (server_id,),
            )
            self.conn.commit()
            
            # Remove from default servers if necessary
            if server["is_default"]:
                self.config.remove_default_mcp_server(server["name"])
                
            return cursor.rowcount > 0
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting MCP server: {str(e)}")
            return False
    
    # Chat history management
    def add_chat_message(
        self,
        user_message: str,
        assistant_message: str,
        cluster_id: Optional[int] = None,
    ) -> int:
        """Add a chat message.
        
        Args:
            user_message: User message.
            assistant_message: Assistant message.
            cluster_id: Optional cluster ID.
            
        Returns:
            Chat ID.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history (user_message, assistant_message, cluster_id)
                VALUES (?, ?, ?)
                """,
                (user_message, assistant_message, cluster_id),
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding chat message: {str(e)}")
            raise
    
    def get_chat_history(
        self, limit: int = 20, cluster_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get chat history.
        
        Args:
            limit: Maximum number of messages to return.
            cluster_id: Optional cluster ID to filter by.
            
        Returns:
            List of chat messages.
        """
        try:
            cursor = self.conn.cursor()
            
            if cluster_id is not None:
                cursor.execute(
                    """
                    SELECT * FROM chat_history
                    WHERE cluster_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (cluster_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM chat_history
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            raise
    
    def add_tool_execution(
        self,
        chat_id: int,
        tool_name: str,
        server_name: str,
        arguments: Dict[str, Any],
        result: Any,
        execution_time: float,
    ) -> int:
        """Add a tool execution record.
        
        Args:
            chat_id: Chat ID.
            tool_name: Tool name.
            server_name: Server name.
            arguments: Tool arguments.
            result: Tool result.
            execution_time: Execution time in seconds.
            
        Returns:
            Tool execution ID.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_executions (
                    chat_id, tool_name, server_name, arguments, result, execution_time
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id, tool_name, server_name,
                    json.dumps(arguments), json.dumps(result),
                    execution_time,
                ),
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding tool execution: {str(e)}")
            raise
    
    def get_tool_executions(
        self, chat_id: Optional[int] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get tool executions.
        
        Args:
            chat_id: Optional chat ID to filter by.
            limit: Maximum number of records to return.
            
        Returns:
            List of tool execution records.
        """
        try:
            cursor = self.conn.cursor()
            
            if chat_id is not None:
                cursor.execute(
                    """
                    SELECT * FROM tool_executions
                    WHERE chat_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM tool_executions
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                
            tool_execs = []
            for row in cursor.fetchall():
                tool_exec = dict(row)
                
                # Deserialize JSON fields
                try:
                    tool_exec["arguments"] = json.loads(tool_exec["arguments"])
                except (json.JSONDecodeError, TypeError):
                    tool_exec["arguments"] = {}
                    
                try:
                    tool_exec["result"] = json.loads(tool_exec["result"])
                except (json.JSONDecodeError, TypeError):
                    tool_exec["result"] = None
                    
                tool_execs.append(tool_exec)
                
            return tool_execs
            
        except Exception as e:
            logger.error(f"Error getting tool executions: {str(e)}")
            raise
    
    def clear_chat_history(self, cluster_id: Optional[int] = None) -> bool:
        """Clear chat history.
        
        Args:
            cluster_id: Optional cluster ID to filter by.
            
        Returns:
            True if the operation was successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            
            if cluster_id is not None:
                cursor.execute(
                    """
                    DELETE FROM chat_history
                    WHERE cluster_id = ?
                    """,
                    (cluster_id,),
                )
            else:
                cursor.execute("DELETE FROM chat_history")
                
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error clearing chat history: {str(e)}")
            return False
    
    # LLM configuration management
    def add_llm_config(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_default: bool = False,
    ) -> int:
        """Add an LLM configuration.
        
        Args:
            provider: LLM provider.
            model: Model name.
            api_key: API key.
            base_url: Base URL.
            parameters: Additional parameters.
            is_default: Whether this is the default configuration.
            
        Returns:
            Configuration ID.
        """
        try:
            # If this is the default, unset existing defaults
            if is_default:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    UPDATE llm_config
                    SET is_default = 0
                    """
                )
                
            # Serialize parameters as JSON
            params_json = json.dumps(parameters) if parameters else None
            
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO llm_config (
                    provider, model, api_key, base_url, parameters, is_default
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    provider, model, api_key, base_url, params_json,
                    1 if is_default else 0,
                ),
            )
            self.conn.commit()
            
            return cursor.lastrowid
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding LLM configuration: {str(e)}")
            raise
    
    def get_llm_config(self, config_id: int) -> Optional[Dict[str, Any]]:
        """Get an LLM configuration by ID.
        
        Args:
            config_id: Configuration ID.
            
        Returns:
            Configuration data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT * FROM llm_config WHERE id = ?
                """,
                (config_id,),
            )
            
            row = cursor.fetchone()
            if not row:
                return None
                
            config = dict(row)
            
            # Deserialize parameters from JSON
            if config["parameters"]:
                try:
                    config["parameters"] = json.loads(config["parameters"])
                except json.JSONDecodeError:
                    config["parameters"] = {}
            else:
                config["parameters"] = {}
                
            return config
            
        except Exception as e:
            logger.error(f"Error getting LLM configuration: {str(e)}")
            raise
    
    def get_all_llm_configs(self) -> List[Dict[str, Any]]:
        """Get all LLM configurations.
        
        Returns:
            List of configuration data.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM llm_config")
            
            configs = []
            for row in cursor.fetchall():
                config = self.get_llm_config(row["id"])
                if config:
                    configs.append(config)
                    
            return configs
            
        except Exception as e:
            logger.error(f"Error getting all LLM configurations: {str(e)}")
            raise
    
    def get_default_llm_config(self) -> Optional[Dict[str, Any]]:
        """Get the default LLM configuration.
        
        Returns:
            Configuration data, or None if not found.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT id FROM llm_config WHERE is_default = 1
                """
            )
            
            row = cursor.fetchone()
            if not row:
                # If no default, get the first configuration
                cursor.execute("SELECT id FROM llm_config LIMIT 1")
                row = cursor.fetchone()
                
            if not row:
                return None
                
            return self.get_llm_config(row["id"])
            
        except Exception as e:
            logger.error(f"Error getting default LLM configuration: {str(e)}")
            raise
    
    def update_llm_config(
        self,
        config_id: int,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_default: Optional[bool] = None,
    ) -> bool:
        """Update an LLM configuration.
        
        Args:
            config_id: Configuration ID.
            provider: New provider.
            model: New model name.
            api_key: New API key.
            base_url: New base URL.
            parameters: New parameters.
            is_default: New default flag.
            
        Returns:
            True if the update was successful, False otherwise.
        """
        try:
            # Get current configuration data
            current = self.get_llm_config(config_id)
            if not current:
                return False
                
            # Update fields
            updates = {}
            if provider is not None:
                updates["provider"] = provider
            if model is not None:
                updates["model"] = model
            if api_key is not None:
                updates["api_key"] = api_key
            if base_url is not None:
                updates["base_url"] = base_url
            if parameters is not None:
                updates["parameters"] = json.dumps(parameters)
                
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
                
            # If setting as default, unset existing defaults
            if is_default:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    UPDATE llm_config
                    SET is_default = 0
                    """
                )
                updates["is_default"] = 1
            elif is_default is not None:
                updates["is_default"] = 1 if is_default else 0
                
            # Update configuration record if needed
            if updates:
                # Build update query
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                query = f"UPDATE llm_config SET {set_clause} WHERE id = ?"
                
                # Execute update
                cursor = self.conn.cursor()
                cursor.execute(
                    query,
                    list(updates.values()) + [config_id],
                )
            
            # Commit transaction
            self.conn.commit()
            
            return True
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error updating LLM configuration: {str(e)}")
            return False
    
    def delete_llm_config(self, config_id: int) -> bool:
        """Delete an LLM configuration.
        
        Args:
            config_id: Configuration ID.
            
        Returns:
            True if the deletion was successful, False otherwise.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                DELETE FROM llm_config WHERE id = ?
                """,
                (config_id,),
            )
            self.conn.commit()
            
            return cursor.rowcount > 0
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting LLM configuration: {str(e)}")
            return False
